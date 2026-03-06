"""
Part 3 Context Extension — Picochat on Modal
=============================================

Trains a picochat (d=12) at reduced sequence length (512), extends to
2048 via continued training, and runs a passkey retrieval eval to compare
both checkpoints' ability to use long-range context.

Two-phase approach:
  Phase 1: Train from scratch at seq_len=512 (Chinchilla-optimal, ~50% of data)
  Phase 2: Copy checkpoint, resume training at seq_len=2048 for 600 more steps

Evaluation:
  - Passkey retrieval: tests whether models can retrieve a 5-digit number
    placed at varying distances from the query. The 512-seq model's attention
    window limits it to 512 tokens back; the 2048-seq model attends to full context.
  - Standard bpb + CORE eval via base_eval.py

Usage
-----
Full pipeline on Modal:
    modal run runs/part3_context_modal.py

Individual stages:
    modal run runs/part3_context_modal.py::stage_data
    modal run runs/part3_context_modal.py::stage_tokenizer
    modal run runs/part3_context_modal.py::stage_pretrain_phase1
    modal run runs/part3_context_modal.py::stage_pretrain_phase2
    modal run runs/part3_context_modal.py::stage_eval

Cost reference (8×H100 at ~$28/hr, eval on 4×H100 at ~$14/hr)
---------------------------------------------------------------
    stage_data + tokenizer       : ~12 min    ~$1
    Phase 1 (512-seq, d=12)      : ~20 min    ~$9
    Phase 2 (2048-seq, d=12)     : ~25 min    ~$12
    Eval (2 models, passkey+std) : ~90 min    ~$21
    Total                                     ~$43
"""

import os
import json
import shutil
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Phase 1: short context training (Chinchilla-optimal)
PHASE1_SEQ_LEN = 512
PHASE1_TAG = "pico_ctx512"
# Phase 1 iterations determined by Chinchilla ratio (~2205 for d=12)

# Phase 2: extended context (continued training)
PHASE2_SEQ_LEN = 2048
PHASE2_TAG = "pico_ctx2048"
PHASE2_EXTRA_STEPS = 600  # additional steps at longer context

# Architecture (same for both phases)
DEPTH = 12
ASPECT_RATIO = 64
HEAD_DIM = 64
WINDOW_PATTERN = "L"

# ── GPU ──────────────────────────────────────────────────────────────────────
GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:4"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1]) if ":" in GPU_TRAIN else 1
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])  if ":" in GPU_EVAL  else 1

# ── Data ─────────────────────────────────────────────────────────────────────
NUM_SHARDS = 80  # ~8GB, sufficient for Chinchilla-optimal d=12

# ── Batch size ───────────────────────────────────────────────────────────────
DEVICE_BATCH_SIZE = 16

# ── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "picochat-context"

# ── Timeouts ─────────────────────────────────────────────────────────────────
PRETRAIN_TIMEOUT_SEC  = 60 * 60 * 2    # 2 hours
EVAL_TIMEOUT_SEC      = 60 * 60 * 4    # 4 hours
DOWNLOAD_TIMEOUT_SEC  = 60 * 90        # 90 min

# ── Volume / cache ───────────────────────────────────────────────────────────
VOLUME_MOUNT   = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR       = "/data/.cache/nanochat"

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

# ── Passkey eval config ──────────────────────────────────────────────────────
PASSKEY_FILLER_LENGTHS = [256, 512, 768, 1024, 1536]  # in tokens
PASSKEY_TRIALS = 50   # per filler length
PASSKEY_SEED = 42


# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-picochat-context")

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

def _python(module: str, args: list | None = None, *, cwd: str = "/root/nanochat") -> None:
    args = args or []
    cmd = f"cd {cwd} && uv run python -m {module} {' '.join(args)}"
    _run(cmd)


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    print(cmd)
    _run(cmd)


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


# =============================================================================
# STAGE 0: DATA DOWNLOAD
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
    """Download FineWeb-EDU dataset shards."""
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()
    print(f"Done: {num_shards} shards downloaded.")


# =============================================================================
# STAGE 1: TOKENIZER TRAINING
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """Train BPE tokenizer on 2B characters of FineWeb-EDU."""
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
# STAGE 2: PHASE 1 — PRETRAIN AT SEQ_LEN=512
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_phase1() -> None:
    """
    Phase 1: Train d=12 picochat at seq_len=512.
    Uses Chinchilla-optimal ratio (--target-param-data-ratio=10.5), which
    produces ~1200 iterations. This trains on ~50% of the 80-shard dataset.
    """
    _setup_cache()
    print("Resetting training report...")
    _python("nanochat.report", ["reset"])
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH}",
            f"--aspect-ratio={ASPECT_RATIO}",
            f"--head-dim={HEAD_DIM}",
            f"--window-pattern={WINDOW_PATTERN}",
            f"--max-seq-len={PHASE1_SEQ_LEN}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=1000",
            f"--model-tag={PHASE1_TAG}",
            f"--run={WANDB_PROJECT}_phase1_ctx512",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("Phase 1 (seq_len=512) complete.")


# =============================================================================
# STAGE 3: PHASE 2 — EXTEND TO SEQ_LEN=2048
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_phase2() -> None:
    """
    Phase 2: Copy phase 1 checkpoint to a new directory, then resume
    training at seq_len=2048 for 600 more steps.

    num_iterations is set to phase1_last_step + 600, so the LR schedule
    warmdown covers the entire training horizon correctly. With
    warmdown_ratio=0.5 and num_iterations = last_step + 600, warmdown
    starts at (last_step+600)/2. Since we resume at last_step, the LR
    is naturally reduced — appropriate for context extension fine-tuning.

    No code changes needed: parameter shapes are identical between 512
    and 2048 sequence lengths (RoPE buffers are non-persistent and
    window sizes are recomputed on init).

    Batch size adjustment:
      512-seq: 16 × 512 × 8 = 65,536 tok/micro-step → 8 grad accum steps
      2048-seq: 16 × 2048 × 8 = 262,144 tok/micro-step → 2 grad accum steps
    Same total batch size (~524K tokens) per optimization step.
    """
    _setup_cache()

    checkpoints_base = os.path.join(NANOCHAT_CACHE, "base_checkpoints")
    phase1_dir = os.path.join(checkpoints_base, PHASE1_TAG)
    phase2_dir = os.path.join(checkpoints_base, PHASE2_TAG)

    # Find the last step in phase 1 checkpoint
    import glob as globmod
    model_files = globmod.glob(os.path.join(phase1_dir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No phase 1 checkpoints found in {phase1_dir}")
    last_step = max(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in model_files)
    num_iterations = last_step + PHASE2_EXTRA_STEPS
    print(f"Phase 1 last step: {last_step}")
    print(f"Phase 2 num_iterations: {num_iterations} ({last_step} + {PHASE2_EXTRA_STEPS})")

    # Copy checkpoint files from phase1 → phase2 directory
    os.makedirs(phase2_dir, exist_ok=True)
    for pattern in [f"model_{last_step:06d}.pt", f"meta_{last_step:06d}.json", f"optim_{last_step:06d}_rank*.pt"]:
        for src in globmod.glob(os.path.join(phase1_dir, pattern)):
            dst = os.path.join(phase2_dir, os.path.basename(src))
            if not os.path.exists(dst):
                print(f"Copying {src} -> {dst}")
                shutil.copy2(src, dst)
            else:
                print(f"Already exists, skipping: {dst}")

    # Patch the meta JSON: update model_config.sequence_len and max_seq_len
    # so that build_model reconstructs with the right window sizes
    meta_src = os.path.join(phase2_dir, f"meta_{last_step:06d}.json")
    with open(meta_src, "r") as f:
        meta = json.load(f)
    meta["model_config"]["sequence_len"] = PHASE2_SEQ_LEN
    meta["max_seq_len"] = PHASE2_SEQ_LEN
    with open(meta_src, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Patched meta JSON: sequence_len={PHASE2_SEQ_LEN}")

    # Resume training at the extended sequence length
    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH}",
            f"--aspect-ratio={ASPECT_RATIO}",
            f"--head-dim={HEAD_DIM}",
            f"--window-pattern={WINDOW_PATTERN}",
            f"--max-seq-len={PHASE2_SEQ_LEN}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            f"--num-iterations={num_iterations}",
            f"--resume-from-step={last_step}",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=1000",
            f"--model-tag={PHASE2_TAG}",
            f"--run={WANDB_PROJECT}_phase2_ctx2048",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("Phase 2 (seq_len=2048) complete.")


# =============================================================================
# STAGE 4: EVAL — PASSKEY RETRIEVAL + STANDARD BPB/CORE
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
    Run passkey retrieval eval (custom) + standard bpb/CORE eval on both models.

    Passkey retrieval:
      - Insert a 5-digit passkey at the start of a prompt, pad with filler
        tokens, then ask the model to recall the passkey.
      - Vary filler lengths: 256, 512, 768, 1024, 1536 tokens.
      - 50 trials per length, deterministic RNG.
      - 512-seq model can only attend 512 tokens back → fails at long distances.
      - 2048-seq model attends to full context → high accuracy everywhere.

    Standard eval: bpb + CORE via base_eval.py (same as Part 2).
    """
    _setup_cache()

    # ── Passkey retrieval eval ───────────────────────────────────────────────
    passkey_output = os.path.join(NANOCHAT_CACHE, "passkey_results.json")
    filler_str = " ".join(str(x) for x in PASSKEY_FILLER_LENGTHS)
    _python("scripts.passkey_eval", [
        f"--model-tags {PHASE1_TAG} {PHASE2_TAG}",
        f"--filler-lengths {filler_str}",
        f"--trials {PASSKEY_TRIALS}",
        f"--seed {PASSKEY_SEED}",
        f"--output {passkey_output}",
    ])
    volume.commit()

    # ── Standard eval (bpb + CORE) ──────────────────────────────────────────
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _curl(EVAL_BUNDLE_URL, zip_path)
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    results = {}
    for tag in [PHASE1_TAG, PHASE2_TAG]:
        print(f"\n{'='*60}\nStandard eval: {tag}\n{'='*60}")
        log_path = os.path.join(NANOCHAT_CACHE, f"{tag}_eval.txt")

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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONTEXT EXTENSION RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'val_bpb':>10} {'CORE':>10}")
    print("-" * 60)
    for tag, r in results.items():
        print(f"{tag:<20} {r['val_bpb']:>10} {r['core']:>10}")
    print("=" * 60)
    print(f"\nPasskey retrieval results logged above.")
    print(f"Full eval logs: {NANOCHAT_CACHE}/<model>_eval.txt")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Full Part 3 context extension pipeline.

        0. Download 80 FineWeb-EDU shards        (CPU,    ~10 min,  ~$0.10)
        1. Train BPE tokenizer                   (1xH100, ~2 min,   ~$0.12)
        2. Phase 1: train at seq_len=512         (8xH100, ~20 min,  ~$9)
        3. Phase 2: extend to seq_len=2048       (8xH100, ~25 min,  ~$12)
        4. Eval: passkey + bpb + CORE            (4xH100, ~90 min,  ~$21)

    Estimated total: ~$43
    """
    w = 64
    print("\n" + "=" * w)
    print("Picochat Context Extension — Part 3")
    print(f"  Phase 1 : d={DEPTH}, seq_len={PHASE1_SEQ_LEN}, tag={PHASE1_TAG}")
    print(f"  Phase 2 : d={DEPTH}, seq_len={PHASE2_SEQ_LEN}, tag={PHASE2_TAG}")
    print(f"  GPU     : {GPU_TRAIN}  |  Shards: {NUM_SHARDS}  |  WandB: {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[0/4] Downloading FineWeb-EDU shards...")
    stage_data.remote(num_shards=NUM_SHARDS)

    print("[1/4] Training tokenizer...")
    stage_tokenizer.remote()

    print("[2/4] Phase 1: Training at seq_len=512...")
    stage_pretrain_phase1.remote()

    print("[3/4] Phase 2: Extending to seq_len=2048...")
    stage_pretrain_phase2.remote()

    print("[4/4] Evaluating both models (passkey + bpb + CORE)...")
    stage_eval.remote()

    print("\n" + "=" * w)
    print("Context extension complete!")
    print("  Checkpoints + eval logs in Modal Volume 'nanochat-vol'")
    print(f"  W&B: https://wandb.ai/your-username/{WANDB_PROJECT}")
    print("=" * w + "\n")
