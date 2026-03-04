"""
Part 2 Ablation Study — Picochat on Modal
==========================================

Trains 4 picochat models and compares them:
  - pico_baseline    : d=8, aspect-ratio=64, ReLU², standard MHA
  - pico_swiglu      : d=8, aspect-ratio=64, SwiGLU only (Shazeer 2020)
  - pico_cla         : d=8, aspect-ratio=64, ReLU², CLA-2 only (Brandon 2024)
  - pico_swiglu_cla  : d=8, aspect-ratio=64, SwiGLU + CLA-2 combined

The combined run tests whether the two changes compound positively, negatively,
or independently — a key question for recommending them together in the full speedrun.

Usage
-----
Smoke test (local CPU/MPS, ~3 min):
    bash runs/runpico_test.sh

Full ablation on Modal (~20 min each, ~$28 total):
    modal run runs/pico_ablation_modal.py

Individual stages:
    modal run runs/pico_ablation_modal.py::stage_data
    modal run runs/pico_ablation_modal.py::stage_tokenizer
    modal run runs/pico_ablation_modal.py::stage_pretrain_baseline
    modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu
    modal run runs/pico_ablation_modal.py::stage_pretrain_cla
    modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu_cla
    modal run runs/pico_ablation_modal.py::stage_eval

Cost reference (8×H100 at ~$31/hr for the node, eval on 4×H100 at ~$14/hr)
---------------------------------------------------------------------------
    stage_data + tokenizer         :  ~10 min   ~$5.30
    pico_baseline      d=12 8×H100 : ~5 min     ~$2.60
    pico_cla           d=12 8×H100 : ~5 min     ~$2.60
    stage_eval (bpb+CORE+sample)   : ~60 min    ~$4.70
    Total                                       ~$15.20

    (swiglu and swiglu_cla stages exist but are not run in the default
     pipeline — call them individually if needed)

Note: d=8 preliminary runs (val_bpb: baseline=1.027, cla=1.050; CORE:
baseline=0.066, cla=0.057) showed CLA hurts at shallow depth. Switched
to d=12 for more meaningful ablation.

Notes
-----
- Data and tokenizer stages are shared across all 4 runs (run once).
- GPU config matches the reference speedrun (H100:8, device-batch-size=16).
- 40 shards (~4GB) is sufficient for Chinchilla-optimal training at d=8.
- W&B tracks all 4 runs under the same project for easy comparison.
- CORE eval runs once per model after training (not during), to avoid
  paying for 30-min evaluations at every checkpoint.
- The eval bundle (~1GB) is downloaded once and cached in the volume.
"""

import os
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Picochat model configs
#   baseline   : d=12, ReLU², standard MHA, per-layer FFN — control model
#   cla        : d=12, ReLU², CLA-2 KV sharing (Brandon 2024) — negative result
#   shared_ffn : d=12, ReLU², one shared MLP across all layers (MobiLlama 2024)
CONFIGS = {
    "pico_baseline":        {"depth": 12, "aspect_ratio": 64, "cla": False, "shared_ffn": False},
    "pico_cla":             {"depth": 12, "aspect_ratio": 64, "cla": True,  "shared_ffn": False},
    "pico_shared_ffn":      {"depth": 12, "aspect_ratio": 64, "cla": False, "shared_ffn": True},
    "pico_cla_shared_ffn":  {"depth": 12, "aspect_ratio": 64, "cla": True,  "shared_ffn": True},
}

# ── GPU ───────────────────────────────────────────────────────────────────────
# "H100:8" = 8 H100s, matches reference speedrun config.
# "H100:4" = 4 H100s for eval (speedrun uses GPU_PRETRAIN for eval too, but
#             4 is sufficient for base_eval and halves eval cost).
GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:4"

# Derive GPU counts dynamically from the GPU string, mirrors reference script
_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1]) if ":" in GPU_TRAIN else 1
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])  if ":" in GPU_EVAL  else 1

# ── Data ──────────────────────────────────────────────────────────────────────
# 80 shards (~8GB) — sufficient for Chinchilla-optimal training at d=12.
# d=12 has ~125M params; Chinchilla at 10.5× ≈ 1.3B tokens ≈ 26 shards needed.
# 80 shards provides comfortable headroom (~1/3 of the 240-shard speedrun).
NUM_SHARDS = 80

# ── Batch size ────────────────────────────────────────────────────────────────
# Matches reference speedrun DEVICE_BATCH_SIZE.
# H100 80GB: 32 fits for d24 per reference; d=8 is much smaller so 32 is safe.
DEVICE_BATCH_SIZE = 16
DEVICE_BATCH_SIZE_DEEP = 16  # d=34 shared_ffn: model_dim=768 still fits at bs=16; activations are small

# ── WandB ─────────────────────────────────────────────────────────────────────
# Set to "dummy" to disable WandB logging
WANDB_PROJECT = "picochat-ablation"

# ── Timeouts ──────────────────────────────────────────────────────────────────
PRETRAIN_TIMEOUT_SEC  = 60 * 60 * 2    # 2 hours (d=8 fits in ~20 min on 4×H100)
EVAL_TIMEOUT_SEC      = 60 * 60 * 4    # 4 hours (CORE ~30 min × 4 models)
DOWNLOAD_TIMEOUT_SEC  = 60 * 90        # 90 min, mirrors reference

# ── Volume / cache ────────────────────────────────────────────────────────────
VOLUME_MOUNT   = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"  # mirrors $NANOCHAT_BASE_DIR
BASE_DIR       = "/data/.cache/nanochat"

# Eval bundle URL (fixed, hosted by Karpathy) — mirrors reference
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


# =============================================================================
# MODAL PRIMITIVES — App, Volume, Secret, Image
# =============================================================================

app = App("nanochat-picochat-ablation")

# Persistent network volume: survives container shutdowns.
# Shares the same volume as the reference speedrun script so data/tokenizer
# are already present if the speedrun was run first.
volume = Volume.from_name("nanochat-vol", create_if_missing=True)

# Secret: injects WANDB_API_KEY and HF_TOKEN as env vars inside containers.
# Create once with:
#   modal secret create nanochat-secrets WANDB_API_KEY=... HF_TOKEN=hf_...
secret = Secret.from_name("nanochat-secrets")

# Container image — mirrors reference script image build exactly
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
# HELPERS — mirrors reference script exactly
# =============================================================================

def _python(module: str, args: list | None = None, *, cwd: str = "/root/nanochat") -> None:
    """Run `python -m {module} [args]` — for non-distributed scripts."""
    args = args or []
    cmd = f"cd {cwd} && uv run python -m {module} {' '.join(args)}"
    _run(cmd)


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    """Run a training script under torchrun for multi-GPU distributed execution."""
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    print(cmd)
    _run(cmd)


def _run(cmd: str) -> None:
    """Shell out to bash, stream stdout/stderr, and raise on failure."""
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


def _setup_cache() -> None:
    """
    Create cache directories and symlink BASE_DIR -> volume.
    Mirrors reference _setup_cache exactly.
    """
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _curl(url: str, dest: str) -> None:
    """Download a file with curl, skipping if already present."""
    if os.path.exists(dest):
        print(f"Already cached, skipping: {dest}")
        return
    _run(f"curl -L -o {dest} {url}")


# =============================================================================
# STAGE 0: DATA DOWNLOAD  (shared, run once)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_data(num_shards: int = NUM_SHARDS) -> None:
    """
    Download FineWeb-EDU dataset shards (CPU-only, run once).
    40 shards (~4GB) is sufficient for Chinchilla-optimal training at d=8.
    Mirrors reference stage_data.
    """
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()
    print(f"Done: {num_shards} shards downloaded.")


# =============================================================================
# STAGE 1: TOKENIZER TRAINING  (shared, run once)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """
    Train a custom BPE tokenizer on 2B characters of FineWeb-EDU.
    Mirrors reference stage_tokenizer exactly.
    """
    _setup_cache()
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer already trained. Skipping tok_train.")
    else:
        print("Training tokenizer on 2B characters...")
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()
    print("Evaluating tokenizer compression ratio...")
    _python("scripts.tok_eval")
    print("Tokenizer ready.")


# =============================================================================
# STAGE 2a: PRETRAIN BASELINE
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_baseline() -> None:
    """
    pico_baseline: d=12, ReLU², standard MHA.
    Control model — no changes from Feb 2026 nanochat defaults.
    Training horizon set by Chinchilla ratio (--target-param-data-ratio=10.5).
    d=12 matches reference quick_test depth (~125M params, 6 CLA pairs).
    """
    _setup_cache()
    print("Resetting training report...")
    _python("nanochat.report", ["reset"])
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--aspect-ratio=64",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--head-dim=64",
            "--window-pattern=L",
            "--core-metric-every=-1",   # CORE runs once after training in stage_eval
            "--sample-every=-1",
            "--save-every=1000",
            "--model-tag=pico_baseline",
            f"--run={WANDB_PROJECT}_baseline",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("pico_baseline complete.")


# =============================================================================
# STAGE 2c: PRETRAIN CLA
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_cla() -> None:
    """
    pico_cla: d=12, CLA-2 cross-layer KV sharing.
    Requires --cla-sharing=2 flag implemented in gpt.py and wired in base_train.py.
    Even-numbered layers reuse K and V from the preceding odd layer.
    All other hyperparameters identical to baseline.
    """
    _setup_cache()
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--aspect-ratio=64",
            "--cla-sharing=2",          # new flag: reuse KV every 2 layers
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--head-dim=64",
            "--window-pattern=L",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=1000",
            "--model-tag=pico_cla",
            f"--run={WANDB_PROJECT}_cla",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("pico_cla complete.")



# =============================================================================
# STAGE 2d: PRETRAIN SHARED FFN
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_shared_ffn() -> None:
    """
    pico_shared_ffn: d=12, one MLP shared across all transformer layers.
    Implements MobiLlama (2024) FFN weight sharing.
    d=12 matches the baseline depth (iso-depth comparison).
    Parameter count: 12 × 2.36M attn + 1 × 4.72M MLP = ~33M (fewer than baseline's 85M).
    Training horizon set by Chinchilla ratio (same as baseline).
    """
    _setup_cache()
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--aspect-ratio=64",
            "--shared-ffn",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--head-dim=64",
            "--window-pattern=L",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=1000",
            "--model-tag=pico_shared_ffn",
            f"--run={WANDB_PROJECT}_shared_ffn",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("pico_shared_ffn complete.")


# =============================================================================
# STAGE 2e: PRETRAIN CLA + SHARED FFN (combined)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_cla_shared_ffn() -> None:
    """
    pico_cla_shared_ffn: d=12, CLA-2 KV sharing + shared FFN combined.
    Both changes at d=12 to match baseline depth.
    CLA-2 applied: 6 sharing pairs at d=12.
    Training horizon set by Chinchilla ratio (same as baseline).
    """
    _setup_cache()
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--aspect-ratio=64",
            "--cla-sharing=2",
            "--shared-ffn",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--head-dim=64",
            "--window-pattern=L",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=1000",
            "--model-tag=pico_cla_shared_ffn",
            f"--run={WANDB_PROJECT}_cla_shared_ffn",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("pico_cla_shared_ffn complete.")


# =============================================================================
# STAGE 3: EVAL ALL MODELS  (bpb + CORE + sample)
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
    Run bpb + CORE + sample eval on all 4 models. Mirrors reference
    stage_post_pretrain_eval but runs base_eval only (covers all three modes).
    CORE is the primary nanochat metric (DCLM benchmark, target 0.256525).
    Eval bundle (~1GB) is downloaded once and cached in the volume.
    Each model takes ~30 min for CORE on 4×H100 (~2 hours total for 4 models).
    """
    _setup_cache()

    # Download eval bundle if not already cached — mirrors reference
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _curl(EVAL_BUNDLE_URL, zip_path)
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    results = {}
    for tag in ["pico_baseline", "pico_cla", "pico_shared_ffn", "pico_cla_shared_ffn"]:
        print(f"\n{'='*60}\nEvaluating {tag}...\n{'='*60}")
        log_path = os.path.join(NANOCHAT_CACHE, f"{tag}_eval.txt")

        # Single torchrun call with tee to capture output
        _run(
            f"cd /root/nanochat && "
            f"uv run torchrun --standalone --nproc_per_node={_N_EVAL_GPUS} "
            f"-m scripts.base_eval -- "
            f"--model-tag {tag} --eval core,bpb,sample "
            f"--device-batch-size=1 --split-tokens=524288 --max-per-task=100 "
            f"2>&1 | tee {log_path}"
        )

        val_bpb, core = "n/a", "n/a"
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if "val bpb:" in line.lower():
                        val_bpb = line.strip().split(":")[-1].strip()
                    if "core metric:" in line.lower():
                        core = line.strip().split(":")[-1].strip()
        results[tag] = {"val_bpb": val_bpb, "core": core}

    volume.commit()

    # Summary table
    print("\n" + "=" * 60)
    print("  PICOCHAT ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'val_bpb':>10} {'CORE':>10}")
    print("-" * 60)
    for tag, r in results.items():
        print(f"{tag:<20} {r['val_bpb']:>10} {r['core']:>10}")
    print("=" * 60)
    print(f"\nFull eval logs: {NANOCHAT_CACHE}/<model>_eval.txt")
    print(f"GPT-2 CORE threshold: 0.256525")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Full Part 2 ablation pipeline. Mirrors reference script stage structure.

        0. Download 40 FineWeb-EDU shards       (CPU,    ~5 min,   ~$0.10)
        1. Train BPE tokenizer                  (1xH100, ~2 min,   ~$0.12)
        2a. Pretrain pico_baseline              (8xH100, ~10 min,  ~$5.20)
        2b. Pretrain pico_cla                   (8xH100, ~10 min,  ~$5.20)
        3. Eval all 4 (bpb + CORE + sample)     (4xH100, ~120 min, ~$9.30)

    Estimated total: ~$30-34 at H100 on-demand pricing (~$3.50/GPU/hr).
    """
    w = 64
    print("\n" + "=" * w)
    print("Picochat Ablation Study — Part 2")
    print(f"  Models : {list(CONFIGS.keys())}")
    print(f"  GPU    : {GPU_TRAIN}  |  Shards: {NUM_SHARDS}  |  WandB: {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[0/5] Downloading FineWeb-EDU shards...")
    stage_data.remote(num_shards=NUM_SHARDS)

    print("[1/5] Training tokenizer...")
    stage_tokenizer.remote()

    print("[2a/5] Training pico_baseline...")
    stage_pretrain_baseline.remote()

    print("[2b/5] Training pico_cla...")
    stage_pretrain_cla.remote()

    print("[2c/5] Training pico_shared_ffn...")
    stage_pretrain_shared_ffn.remote()

    print("[2d/5] Training pico_cla_shared_ffn (combined)...")
    stage_pretrain_cla_shared_ffn.remote()

    print("[3/5] Evaluating all 4 models (bpb + CORE + sample)...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("Ablation complete!")
    print("  Checkpoints + eval logs in Modal Volume 'nanochat-vol'")
    print(f"  W&B: https://wandb.ai/your-username/{WANDB_PROJECT}")
    print("=" * w + "\n")
