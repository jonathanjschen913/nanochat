"""
Part 2: SFT & Midtraining — Modal Training Script
===================================================

Runs SFT experiments on the pretrained nanochat d=26 (differential attention) model:
  Run 1: Baseline (original SFT mixture)
  Run 2: Baseline + MetaMathQA (395K)
  Run 3: Baseline + Orca-Math (200K)
  Run 4: Baseline + DART-Math-Hard (585K)
  Run 5: Baseline + (top 2 from runs 2-4)  — run after analysis

Then evaluates all SFT checkpoints with cache-free generative eval.

Usage
-----
Full pipeline (runs 1-4 + eval):
    modal run runs/part2_sft_modal.py

Individual stages:
    modal run runs/part2_sft_modal.py::stage_sft_baseline
    modal run runs/part2_sft_modal.py::stage_sft_metamathqa
    modal run runs/part2_sft_modal.py::stage_sft_orcamath
    modal run runs/part2_sft_modal.py::stage_sft_dartmath
    modal run runs/part2_sft_modal.py::stage_sft_combo
    modal run runs/part2_sft_modal.py::stage_eval

Cost estimate (8×H100 at ~$28/hr, eval on 4×H100 at ~$14/hr)
--------------------------------------------------------------
    SFT baseline   (~1.17M rows) : ~30 min  ~\\$14
    SFT +MetaMath  (~1.57M rows) : ~40 min  ~\\$19
    SFT +Orca      (~1.37M rows) : ~35 min  ~\\$16
    SFT +DART      (~1.75M rows) : ~45 min  ~\\$21
    SFT combo      (TBD)         : ~50 min  ~\\$23
    Eval (5 models, cache-free)  : ~150 min ~\\$35
    Total                                   ~\\$128
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pretrained checkpoint to load (from Part 4)
MODEL_TAG = "nanochat_d26_diff_attn"

# GPU config
GPU_TRAIN = "H100:8"
GPU_EVAL = "H100:4"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1])
_N_EVAL_GPUS = int(GPU_EVAL.split(":")[1])

# SFT batch size — d=26 needs smaller batch to avoid OOM
DEVICE_BATCH_SIZE = 8

# WandB
WANDB_PROJECT = "nanochat-part2"

# Timeouts
SFT_TIMEOUT_SEC = 60 * 60 * 2     # 2 hours per SFT run
EVAL_TIMEOUT_SEC = 60 * 60 * 4    # 4 hours for all evals

# Volume / paths
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

IDENTITY_CONV_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

# SFT tags for each run
SFT_TAGS = {
    "baseline": "sft_baseline",
    "metamathqa": "sft_metamathqa",
    "orcamath": "sft_orcamath",
    "dartmath": "sft_dartmath",
    "combo": "sft_combo",
}


# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-part2")

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

def _python(module: str, args: list | None = None) -> None:
    args = args or []
    _run(f"cd /root/nanochat && uv run python -m {module} {' '.join(args)}")


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


def _curl(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"Already cached, skipping: {dest}")
        return
    _run(f"curl -L -o {dest} {url}")


def _sft_common_setup() -> None:
    """Common setup for all SFT stages: cache + identity conversations."""
    _setup_cache()
    identity_path = os.path.join(BASE_DIR, "identity_conversations.jsonl")
    _curl(IDENTITY_CONV_URL, identity_path)


def _run_sft(sft_tag: str, extra_args: list | None = None) -> None:
    """Run an SFT training job with common args + optional extras."""
    _sft_common_setup()

    # Delete existing checkpoint to force clean retrain
    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", sft_tag)
    if os.path.exists(sft_dir):
        import shutil
        shutil.rmtree(sft_dir)
        print(f"Deleted existing SFT checkpoint: {sft_dir}")

    args = [
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--run={WANDB_PROJECT}_{sft_tag}",
        f"--model-tag={MODEL_TAG}",
        f"--sft-tag={sft_tag}",
        "--chatcore-every=-1",  # skip in-training eval (diff_attn + KV cache issue)
    ]
    if extra_args:
        args.extend(extra_args)

    _torchrun("scripts.chat_sft", args, nproc=_N_TRAIN_GPUS)
    volume.commit()
    print(f"SFT complete: {sft_tag}")


# =============================================================================
# STAGE: SFT BASELINE (Run 1)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft_baseline() -> None:
    """Run 1: Baseline SFT with original mixture (SmolTalk + MMLU + GSM8K + Spelling)."""
    _run_sft(SFT_TAGS["baseline"])


# =============================================================================
# STAGE: SFT + MetaMathQA (Run 2)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft_metamathqa() -> None:
    """Run 2: Baseline + MetaMathQA (395K math QA pairs with CoT)."""
    _run_sft(SFT_TAGS["metamathqa"], ["--metamathqa"])


# =============================================================================
# STAGE: SFT + Orca-Math (Run 3)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft_orcamath() -> None:
    """Run 3: Baseline + Orca-Math (200K word problems, small-model-optimized)."""
    _run_sft(SFT_TAGS["orcamath"], ["--orcamath"])


# =============================================================================
# STAGE: SFT + DART-Math-Hard (Run 4)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft_dartmath() -> None:
    """Run 4: Baseline + DART-Math-Hard (585K hard math problems with CoT)."""
    _run_sft(SFT_TAGS["dartmath"], ["--dartmath"])


# =============================================================================
# STAGE: SFT COMBO (Run 5) — run after analyzing runs 2-4
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft_combo() -> None:
    """Run 5: Baseline + top 2 math datasets. Edit flags below after reviewing runs 2-4."""
    # TODO: Update these flags based on results from runs 2-4
    _run_sft(SFT_TAGS["combo"], ["--metamathqa", "--dartmath"])


# =============================================================================
# STAGE: EVAL — all SFT checkpoints
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=EVAL_TIMEOUT_SEC,
)
def stage_eval() -> None:
    """
    Evaluate all SFT checkpoints on ChatCORE tasks.
    Uses cache-free generation for generative tasks (diff_attn).
    """
    _setup_cache()

    # Find which SFT checkpoints exist
    chatsft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints")
    available_tags = []
    for tag in SFT_TAGS.values():
        tag_dir = os.path.join(chatsft_dir, tag)
        if os.path.isdir(tag_dir):
            available_tags.append(tag)
            print(f"Found checkpoint: {tag}")
        else:
            print(f"Skipping (not found): {tag}")

    if not available_tags:
        print("No SFT checkpoints found! Run SFT stages first.")
        return

    results = {}
    for tag in available_tags:
        print(f"\n{'='*60}")
        print(f"Evaluating: {tag}")
        print(f"{'='*60}")

        # Run chat_eval on all tasks with max_problems=50 for generative tasks
        _torchrun(
            "scripts.chat_eval",
            [
                "-i", "sft",
                "-g", tag,
                "--max-problems=50",
                f"--batch-size={DEVICE_BATCH_SIZE}",
            ],
            nproc=_N_EVAL_GPUS,
        )

    volume.commit()
    print("\nAll evaluations complete. Check W&B and report for results.")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Run SFT experiments 1-4, then evaluate all checkpoints.

    Run 5 (combo) should be run separately after analyzing results:
        modal run runs/part2_sft_modal.py::stage_sft_combo
        modal run runs/part2_sft_modal.py::stage_eval
    """
    w = 64
    print("\n" + "=" * w)
    print("Part 2: SFT & Midtraining Experiments")
    print(f"  Pretrained model : {MODEL_TAG}")
    print(f"  GPU              : {GPU_TRAIN}")
    print(f"  WandB            : {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[1/5] SFT Baseline...")
    stage_sft_baseline.remote()

    print("[2/5] SFT + MetaMathQA...")
    stage_sft_metamathqa.remote()

    print("[3/5] SFT + Orca-Math...")
    stage_sft_orcamath.remote()

    print("[4/5] SFT + DART-Math-Hard...")
    stage_sft_dartmath.remote()

    print("[5/5] Evaluating all checkpoints...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("Part 2 runs 1-4 complete!")
    print("Review results, then run combo (run 5):")
    print("  modal run runs/part2_sft_modal.py::stage_sft_combo")
    print("  modal run runs/part2_sft_modal.py::stage_eval")
    print("=" * w + "\n")
