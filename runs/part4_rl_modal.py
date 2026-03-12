"""
Part 4: RL on GSM8K with custom reward functions — Modal Training Script
========================================================================

Extends Part 3 baseline with additional reward systems (format, tolerance,
steps, combined) to improve GSM8K performance via richer training signal.

Set the REWARD_FN environment variable to select a reward function:
    REWARD_FN=combined modal run runs/part4_rl_modal.py

Defaults to "combined" if not set.

Usage
-----
Full pipeline (combined reward):
    PYTHONUTF8=1 modal run runs/part4_rl_modal.py

Single reward:
    REWARD_FN=tolerance PYTHONUTF8=1 modal run runs/part4_rl_modal.py

Budget-friendly (cap training steps):
    MAX_STEPS=60 REWARD_FN=combined PYTHONUTF8=1 modal run runs/part4_rl_modal.py

"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Reward function — read from env, default to "combined"
REWARD_FN = os.environ.get("REWARD_FN", "combined")
assert REWARD_FN in ("binary", "format", "tolerance", "steps", "combined"), \
    f"Unknown REWARD_FN='{REWARD_FN}', choose from: binary, format, tolerance, steps, combined"

# Optional: cap training steps (useful for budget-constrained runs)
MAX_STEPS = os.environ.get("MAX_STEPS", None)
if MAX_STEPS is not None:
    MAX_STEPS = int(MAX_STEPS)

# SFT checkpoint to start RL from (best from Part 2, same as Part 3)
SFT_TAG = "sft_combo"

# RL checkpoint output tag — each reward fn saves to its own directory
RL_TAG = f"rl_part4_{REWARD_FN}"

# GPU config
GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:8"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1])
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])

# RL hyperparameters — match Part 3 / Karpathy defaults
DEVICE_BATCH_SIZE  = 8
EXAMPLES_PER_STEP  = 16
NUM_SAMPLES        = 16
MAX_NEW_TOKENS     = 256
TEMPERATURE        = 1.0
TOP_K              = 50
NUM_EPOCHS         = 1

# W&B
WANDB_PROJECT = "nanochat-part4"

# Timeouts
RL_TIMEOUT_SEC   = 60 * 60 * 24  # 24 hours
EVAL_TIMEOUT_SEC = 60 * 60 * 4   # 4 hours

# Volume / paths
VOLUME_MOUNT  = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR       = "/data/.cache/nanochat"

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-part4")

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


def _run(cmd: str) -> None:
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
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
# STAGE: RL TRAINING with custom reward function
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=RL_TIMEOUT_SEC,
)
def stage_rl_train() -> None:
    """
    RL on GSM8K starting from sft_combo with the selected reward function.
    Saves checkpoint to chatrl_checkpoints/{RL_TAG}/ (not sft_combo, to avoid overwrites).
    Logs reward curve and pass@k to W&B project nanochat-part4.
    """
    _setup_cache()

    # Verify the Part 2 SFT checkpoint exists
    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", SFT_TAG)
    if not os.path.isdir(sft_dir):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_dir}\n"
            f"Run Part 2 first: modal run runs/part2_sft_modal.py::stage_sft_combo"
        )
    print(f"Found SFT checkpoint: {sft_dir}")
    print(f"Reward function: {REWARD_FN}")
    print(f"Output tag: {RL_TAG}")

    args = [
        f"--run=rl-part4-{REWARD_FN}",
        f"--model-tag={SFT_TAG}",
        f"--output-tag={RL_TAG}",
        f"--reward-fn={REWARD_FN}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--examples-per-step={EXAMPLES_PER_STEP}",
        f"--num-samples={NUM_SAMPLES}",
        f"--max-new-tokens={MAX_NEW_TOKENS}",
        f"--temperature={TEMPERATURE}",
        f"--top-k={TOP_K}",
        f"--num-epochs={NUM_EPOCHS}",
        "--eval-every=999",
        "--eval-examples=400",
        "--save-every=60",
    ]
    if MAX_STEPS is not None:
        args.append(f"--max-steps={MAX_STEPS}")
    _torchrun("scripts.chat_rl", args, nproc=_N_TRAIN_GPUS)
    volume.commit()
    print(f"RL training complete (reward_fn={REWARD_FN}). Checkpoint saved to chatrl_checkpoints/{RL_TAG}/")


# =============================================================================
# STAGE: EVAL
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=EVAL_TIMEOUT_SEC,
)
def stage_eval() -> None:
    """Evaluate the Part 4 RL checkpoint on GSM8K test set."""
    _setup_cache()

    # Eval loads from chatrl_checkpoints/{RL_TAG}/ via -g flag
    print(f"\n--- pass@1 greedy (temperature=0) — reward_fn={REWARD_FN} ---")
    _torchrun("scripts.chat_eval", [
        "-i", "rl",
        "-g", RL_TAG,
        "-a", "GSM8K",
        "--max-problems=1319",
        f"--batch-size={DEVICE_BATCH_SIZE}",
        "--temperature=0.0",
        "--num-samples=1",
    ], nproc=_N_EVAL_GPUS)

    print(f"\n--- pass@8 sampled (temperature=1) — reward_fn={REWARD_FN} ---")
    _torchrun("scripts.chat_eval", [
        "-i", "rl",
        "-g", RL_TAG,
        "-a", "GSM8K",
        "--max-problems=400",
        f"--batch-size={DEVICE_BATCH_SIZE}",
        "--temperature=1.0",
        "--num-samples=8",
    ], nproc=_N_EVAL_GPUS)

    volume.commit()


# =============================================================================
# STAGE: COLLECT COMPLETIONS (for error analysis / writeup)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=EVAL_TIMEOUT_SEC,
)
def stage_collect_completions() -> None:
    """Collect greedy completions on all 1319 GSM8K test problems for error analysis."""
    _setup_cache()
    out_path = os.path.join(BASE_DIR, f"part4_completions_{REWARD_FN}.jsonl")
    _torchrun("scripts.collect_completions", [
        f"--model-tag={RL_TAG}",
        "--source=rl",
        f"--output={out_path}",
        "--temperature=0.0",
        "--max-new-tokens=256",
    ], nproc=_N_EVAL_GPUS)
    volume.commit()
    print(f"Completions saved to {out_path}")


# =============================================================================
# STAGE: DOWNLOAD COMPLETIONS to local machine
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 5,
)
def stage_read_completions() -> bytes:
    """Return completions file contents so the local entrypoint can save it."""
    _setup_cache()
    path = os.path.join(BASE_DIR, f"part4_completions_{REWARD_FN}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Completions not found at {path}. Run stage_collect_completions first.")
    with open(path, "rb") as f:
        return f.read()


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    w = 64
    print("\n" + "=" * w)
    print("Part 4: RL on GSM8K — Custom Reward Functions")
    print(f"  Reward function    : {REWARD_FN}")
    print(f"  SFT starting point : {SFT_TAG}")
    print(f"  RL output tag      : {RL_TAG}")
    print(f"  Max steps          : {MAX_STEPS or 'full epoch (467)'}")
    print(f"  GPU                : {GPU_TRAIN}")
    print(f"  W&B run name       : rl-part4-{REWARD_FN}")
    print("=" * w + "\n")

    print(f"[1/4] RL training with reward_fn={REWARD_FN}...")
    stage_rl_train.remote()

    print("[2/4] Collecting completions for error analysis...")
    stage_collect_completions.remote()

    print("[3/4] Final eval...")
    stage_eval.remote()

    print("[4/4] Downloading completions locally...")
    data = stage_read_completions.remote()
    local_path = f"dev/part4_completions_{REWARD_FN}.jsonl"
    with open(local_path, "wb") as f:
        f.write(data)
    print(f"Completions saved locally to: {local_path}")

    print("\n" + "=" * w)
    print(f"Done. Check W&B project '{WANDB_PROJECT}' for run 'rl-part4-{REWARD_FN}'.")
    print(f"Completions: {local_path}")
    print("=" * w + "\n")
