"""
Part 3: RL on GSM8K — Modal Training Script
============================================

Replicates Karpathy's GSM8K RL run starting from our best Part 2 SFT checkpoint
(sft_combo: baseline + MetaMathQA + DART-Math-Hard, 20% GSM8K before RL).

Usage
-----
Full pipeline:
    modal run runs/part3_rl_modal.py

Individual stages:
    modal run runs/part3_rl_modal.py::stage_rl_baseline
    modal run runs/part3_rl_modal.py::stage_collect_completions

Download completions locally after collection:
    modal volume get nanochat-vol nanochat_cache/part3_completions.jsonl dev/part3_completions.jsonl

Cost estimate (8×H100 at ~$28/hr)
-----------------------------------
    RL training (467 steps, no mid-eval) : ~4 hrs   ~$112
    Eval + completion collection         : ~60 min  ~$28
    Total                                           ~$140
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

SFT_TAG = "sft_combo"  # Part 2 best checkpoint (20% GSM8K)

GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:8"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1])
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])

DEVICE_BATCH_SIZE  = 8
EXAMPLES_PER_STEP  = 16
NUM_SAMPLES        = 16
MAX_NEW_TOKENS     = 256
TEMPERATURE        = 1.0
TOP_K              = 50
NUM_EPOCHS         = 1

# NOTE: chat_rl.py hardcodes project="nanochat-rl" — only the run name uses this value
WANDB_PROJECT = "nanochat-part3"

RL_TIMEOUT_SEC   = 60 * 60 * 24  # 24 hours
EVAL_TIMEOUT_SEC = 60 * 60 * 4   # 4 hours

VOLUME_MOUNT   = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR       = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-part3")

volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(local_path=".", remote_path="/root/nanochat", copy=True,
                   ignore=[".venv", "__pycache__", "*.pyc", ".git"])
    .workdir("/root/nanochat")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> $HOME/.bashrc",
        "bash -c 'source $HOME/.cargo/env'",
    )
    .pip_install("uv")
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": BASE_DIR,
        "HF_HOME": "/data/.cache/huggingface",
    })
    .run_commands(
        "cd /root/nanochat && uv sync --extra gpu --no-install-project",
    )
)


# =============================================================================
# HELPERS
# =============================================================================

def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"TORCHELASTIC_ERROR_FILE=/tmp/torcherror.json "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    print(cmd)
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        subprocess.run(["bash", "-c", "cat /tmp/torcherror.json 2>/dev/null || true"])
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


def _setup_cache() -> None:
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


# =============================================================================
# STAGE: RL TRAINING
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=RL_TIMEOUT_SEC,
)
def stage_rl_baseline() -> None:
    """
    RL on GSM8K starting from sft_combo with binary reward.
    Replicates Karpathy's default chat_rl.py settings.
    Checkpoint saved to chatrl_checkpoints/sft_combo/.
    """
    _setup_cache()

    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", SFT_TAG)
    if not os.path.isdir(sft_dir):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_dir}\n"
            f"Run Part 2 first: modal run runs/part2_sft_modal.py::stage_sft_combo"
        )
    print(f"Found SFT checkpoint: {sft_dir}")

    _torchrun("scripts.chat_rl", [
        f"--run={WANDB_PROJECT}_rl_baseline",
        f"--model-tag={SFT_TAG}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--examples-per-step={EXAMPLES_PER_STEP}",
        f"--num-samples={NUM_SAMPLES}",
        f"--max-new-tokens={MAX_NEW_TOKENS}",
        f"--temperature={TEMPERATURE}",
        f"--top-k={TOP_K}",
        f"--num-epochs={NUM_EPOCHS}",
        "--eval-every=999",
        "--save-every=60",
    ], nproc=_N_TRAIN_GPUS)
    volume.commit()
    print(f"RL training complete. Checkpoint: chatrl_checkpoints/{SFT_TAG}/")


# =============================================================================
# STAGE: COLLECT COMPLETIONS + EVAL (combined)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=EVAL_TIMEOUT_SEC,
)
def stage_collect_completions() -> None:
    """
    Combined eval + completion collection:
    - Collects greedy completions on all 1319 GSM8K test problems (8 GPUs)
    - Reports pass@1 greedy accuracy (comparable to Karpathy's 7.58%)
    - Runs pass@8 sampled eval for upper bound
    Saves completions to BASE_DIR/part3_completions.jsonl.
    Download locally: modal volume get nanochat-vol nanochat_cache/part3_completions.jsonl dev/part3_completions.jsonl
    """
    _setup_cache()

    out_path = os.path.join(BASE_DIR, "part3_completions.jsonl")

    print("\n--- pass@1 greedy + collecting completions (1319 problems, 8 GPUs) ---")
    _torchrun("scripts.collect_completions", [
        f"--model-tag={SFT_TAG}",
        "--source=rl",
        f"--output={out_path}",
        "--temperature=0.0",
        "--max-new-tokens=256",
    ], nproc=_N_EVAL_GPUS)

    print("\n--- pass@8 sampled (temperature=1, 400 problems) ---")
    _torchrun("scripts.chat_eval", [
        "-i", "rl",
        "-g", SFT_TAG,
        "-a", "GSM8K",
        "--max-problems=400",
        f"--batch-size={DEVICE_BATCH_SIZE}",
        "--temperature=1.0",
        "--num-samples=8",
    ], nproc=_N_EVAL_GPUS)

    volume.commit()
    print(f"\nKarpathy after RL (pass@1 greedy): 7.58%")
    print(f"Our SFT baseline (pass@1 greedy) : 20.0%")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    w = 64
    print("\n" + "=" * w)
    print("Part 3: RL on GSM8K — Replication Run")
    print(f"  SFT starting point : {SFT_TAG} (20% GSM8K from Part 2)")
    print(f"  GPU                : {GPU_TRAIN}")
    print(f"  W&B run            : {WANDB_PROJECT}_rl_baseline  (project: nanochat-rl)")
    print("=" * w + "\n")

    print("[1/2] RL training...")
    stage_rl_baseline.remote()

    print("[2/2] Eval + collecting completions...")
    stage_collect_completions.remote()

    print("\n" + "=" * w)
    print("Part 3 complete!")
    print("Download completions:")
    print("  modal volume get nanochat-vol nanochat_cache/part3_completions.jsonl dev/part3_completions.jsonl")
    print("Run analysis:")
    print("  python dev/part3_analysis.py")
    print("=" * w + "\n")
