"""
Part 4: Final Nanochat — Modal Training Script
===============================================

Trains a full nanochat model at d=26 with Differential Attention (Ye et al., ICLR 2025)
and evaluates it against the developer's baseline nanochat results.

Also trains a picochat baseline (d=12) + SFT on this volume for the emergent abilities
comparison in Part 4 (picochat answers vs nanochat answers side-by-side).

Usage
-----
Full pipeline:
    modal run runs/nanochat_modal.py

Individual stages:
    modal run runs/nanochat_modal.py::stage_data
    modal run runs/nanochat_modal.py::stage_tokenizer
    modal run runs/nanochat_modal.py::stage_pretrain_pico
    modal run runs/nanochat_modal.py::stage_sft_pico
    modal run runs/nanochat_modal.py::stage_pretrain
    modal run runs/nanochat_modal.py::stage_sft
    modal run runs/nanochat_modal.py::stage_eval
    modal run runs/nanochat_modal.py::stage_emergent_abilities

Cost reference (8×H100 at ~$28/hr, eval on 4×H100 at ~$14/hr)
---------------------------------------------------------------
    stage_data + tokenizer          : ~10 min    ~$5.30
    stage_pretrain_pico  d=12 diff_attn 8×H100: ~40 min    ~$18.70
    stage_sft_pico       d=12 diff_attn 8×H100: ~10 min    ~$4.70
    stage_pretrain  d=26 8×H100     : ~3 hrs     ~$84.00
    stage_sft       d=26 8×H100     : ~30 min    ~$14.00
    stage_eval      bpb+CORE 4×H100 : ~30 min    ~$7.00
    Total                                        ~$134
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model tag used for checkpoint paths on disk
MODEL_TAG = "nanochat_d26_diff_attn"

# GPU config — matches the reference speedrun
GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:4"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1])
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])

# Data — 370 shards matches the reference speedrun for d=26 (needs ~10B tokens)
NUM_SHARDS = 370

# Batch size — matches reference speedrun for d=26
DEVICE_BATCH_SIZE = 16

# WandB
WANDB_PROJECT = "nanochat-part4"

# Timeouts
PRETRAIN_TIMEOUT_SEC = 60 * 60 * 5   # 5 hours (d=26 ~3 hrs + buffer)
SFT_TIMEOUT_SEC      = 60 * 60 * 2   # 2 hours
EVAL_TIMEOUT_SEC     = 60 * 60 * 2   # 2 hours
DOWNLOAD_TIMEOUT_SEC = 60 * 90       # 90 min

# Volume / paths
VOLUME_MOUNT   = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR       = "/data/.cache/nanochat"

EVAL_BUNDLE_URL      = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
IDENTITY_CONV_URL    = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"


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
        # Print error file for diagnosis before raising
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
    """Download FineWeb-EDU shards. 370 shards needed for ~10B tokens at d=26."""
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards...")
    _python("nanochat.dataset", ["-n", str(num_shards)])
    volume.commit()
    print(f"Done: {num_shards} shards downloaded.")


# =============================================================================
# STAGE 1: TOKENIZER
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """Train BPE tokenizer (shared with Part 2; skipped if already present)."""
    _setup_cache()
    tokenizer_dir = os.path.join(NANOCHAT_CACHE, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        print("Tokenizer already trained. Skipping.")
    else:
        print("Training tokenizer on 2B characters...")
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()
    _python("scripts.tok_eval")
    print("Tokenizer ready.")


# =============================================================================
# STAGE 2a: PRETRAIN — picochat baseline d=12 (for emergent abilities comparison)
# =============================================================================

PICO_TAG = "pico_diff_attn"

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=60 * 90,  # 90 min buffer for d=12
)
def stage_pretrain_pico() -> None:
    """
    Pretrain picochat with differential attention at d=12 (Chinchilla-optimal).
    Same architecture as nanochat but at smaller scale — enables a clean
    scale-only comparison for the Part 4 emergent abilities and scaling law.
    """
    _setup_cache()
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--aspect-ratio=64",
            "--head-dim=64",
            "--window-pattern=L",
            "--differential-attn",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--core-metric-every=-1",
            "--sample-every=-1",
            f"--model-tag={PICO_TAG}",
            f"--run={WANDB_PROJECT}_{PICO_TAG}",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Picochat pretrain complete: {PICO_TAG}")


# =============================================================================
# STAGE 2b: SFT — picochat baseline (for emergent abilities comparison)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=60 * 60,
)
def stage_sft_pico() -> None:
    """SFT fine-tuning on the picochat diff_attn checkpoint."""
    _setup_cache()
    identity_path = os.path.join(BASE_DIR, "identity_conversations.jsonl")
    _curl(IDENTITY_CONV_URL, identity_path)
    _torchrun(
        "scripts.chat_sft",
        [
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            f"--run={WANDB_PROJECT}_{PICO_TAG}_sft",
            f"--model-tag={PICO_TAG}",
            "--chatcore-every=-1",  # diff_attn incompatible with KV cache eval
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("Picochat SFT complete.")


# =============================================================================
# STAGE 2c: PRETRAIN — nanochat d=26 + Differential Attention
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain() -> None:
    """
    Pretrain nanochat at d=26 with Differential Attention.

    Config mirrors the developer's leaderboard Run 3 exactly, with --differential-attn added:
      --depth=26                     : full nanochat scale (~830M params)
      --target-param-data-ratio=8.25 : slightly undertrained vs Chinchilla (10.5x),
                                       tuned to hit GPT-2 CORE threshold at d=26
      --device-batch-size=16         : OOM at 32 for d=26; gradient accumulation compensates
      --fp8                          : fp8 training for speed (H100 required)
      --differential-attn            : Part 2 architecture change with smallest bpb penalty
    """
    _setup_cache()
    _python("nanochat.report", ["reset"])
    _torchrun(
        "scripts.base_train",
        [
            "--depth=26",
            "--head-dim=64",                 # 26*64=1664 dim → 26 heads (even, required for diff attn)
            "--target-param-data-ratio=8.25",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            "--total-batch-size=1048576",    # 1M token batch — matches leaderboard Run 3
            "--fp8",
            "--differential-attn",
            "--core-metric-every=-1",        # disable CORE during training; runs in stage_eval
            "--core-metric-max-per-task=-1", # full CORE eval (not sampled)
            "--sample-every=-1",             # no periodic sampling
            "--save-every=-1",               # no intermediate checkpoints (saves disk)
            f"--model-tag={MODEL_TAG}",
            f"--run={WANDB_PROJECT}_{MODEL_TAG}",
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print(f"Pretraining complete: {MODEL_TAG}")


# =============================================================================
# STAGE 3: SFT FINE-TUNING
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    timeout=SFT_TIMEOUT_SEC,
)
def stage_sft() -> None:
    """
    SFT fine-tuning on the pretrained nanochat checkpoint.
    Required for the emergent abilities evaluation — the base model does not
    follow the chat format needed for question-answering demos.
    Downloads identity_conversations.jsonl (synthetic personality data) then runs SFT.
    """
    _setup_cache()

    # Delete existing SFT checkpoint to force clean retrain at bs=8
    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", MODEL_TAG)
    if os.path.exists(sft_dir):
        import shutil
        shutil.rmtree(sft_dir)
        print(f"Deleted existing SFT checkpoint: {sft_dir}")

    # Download identity conversations for SFT personality
    identity_path = os.path.join(BASE_DIR, "identity_conversations.jsonl")
    _curl(IDENTITY_CONV_URL, identity_path)

    _torchrun(
        "scripts.chat_sft",
        [
            "--device-batch-size=8",  # d=26 needs smaller batch for SFT to avoid OOM
            f"--run={WANDB_PROJECT}_{MODEL_TAG}_sft",
            f"--model-tag={MODEL_TAG}",
            "--chatcore-every=-1",  # diff_attn incompatible with KV cache eval
        ],
        nproc=_N_TRAIN_GPUS,
    )
    volume.commit()
    print("SFT complete.")


# =============================================================================
# STAGE 4: EVAL — bpb + CORE
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
    Run bpb + CORE eval on the pretrained nanochat checkpoint.
    Note: KV cache inference is not supported with differential attention,
    so sample eval is skipped (diff_attn raises NotImplementedError in engine.py).
    """
    _setup_cache()

    # Download eval bundle if not already cached
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _curl(EVAL_BUNDLE_URL, zip_path)
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    log_path = os.path.join(NANOCHAT_CACHE, f"{MODEL_TAG}_eval.txt")

    # Eval modes: skip 'sample' — KV cache not supported for differential attention
    _run(
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={_N_EVAL_GPUS} "
        f"-m scripts.base_eval -- "
        f"--model-tag {MODEL_TAG} --eval core,bpb "
        f"--device-batch-size=16 "
        f"--max-per-task=-1 "
        f"2>&1 | tee {log_path}"
    )

    volume.commit()

    # Parse and print summary
    val_bpb, core = "n/a", "n/a"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "val bpb:" in line.lower():
                    val_bpb = line.strip().split(":")[-1].strip()
                if "core metric:" in line.lower():
                    core = line.strip().split(":")[-1].strip()

    print("\n" + "=" * 60)
    print("  NANOCHAT D=26 + DIFFERENTIAL ATTENTION — RESULTS")
    print("=" * 60)
    print(f"  Model tag : {MODEL_TAG}")
    print(f"  val_bpb   : {val_bpb}")
    print(f"  CORE      : {core}")
    print(f"  GPT-2 threshold: 0.256525")
    print("=" * 60)
    print(f"\nFull eval log: {log_path}")


# =============================================================================
# STAGE 5: EMERGENT ABILITIES
# =============================================================================

EMERGENT_QUESTIONS = [
    "What color is the sky?",
    "How many days are in a week?",
    "What is 2 plus 2?",
    "What is the opposite of hot?",
    "How many sides does a triangle have?",
    "What do plants need to grow?",
    "What is water made of?",
    "What is the boiling point of water?",
    "What is the largest planet in our solar system?",
    "What do humans need to survive?",
]

EMERGENT_SCRIPT = '''
import sys, torch
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import get_base_dir
import os

tag = sys.argv[1]
questions = sys.argv[2:]
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, "chatsft_checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, _ = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=tag)
model = model.to(dtype=torch.bfloat16)
model.eval()

USER_START = "<|user_start|>"
ASST_START = "<|assistant_start|>"

for q in questions:
    ids = (
        [tokenizer.encode_special(USER_START)]
        + tokenizer.encode(q)
        + [tokenizer.encode_special(ASST_START)]
    )
    ids = torch.tensor([ids], device=device)
    stop_ids = set()
    for s in ["<|user_start|>", "<|end|>", "<|assistant_start|>", "<|assistant_end|>"]:
        try:
            tok_id = tokenizer.encode_special(s)
            stop_ids.add(tok_id)
        except Exception:
            pass
    with torch.no_grad():
        for _ in range(200):
            logits = model(ids, kv_cache=None)
            next_tok = logits[0, -1].argmax().item()
            if next_tok in stop_ids:
                break
            ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)
    response = tokenizer.decode(ids[0].tolist())
    response = response.split(ASST_START)[-1].strip()
    print(f"\\nQ: {q}\\nA: {response}\\n")
'''

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_emergent_abilities() -> None:
    """
    Run all 10 emergent ability questions on both picochat and nanochat SFT checkpoints.
    Uses simple greedy decode without KV cache (required for differential attention).
    Requires stage_sft_pico and stage_sft to have completed first.
    """
    _setup_cache()
    # Write inline script to disk
    script_path = "/root/nanochat/emergent_eval.py"
    with open(script_path, "w") as f:
        f.write(EMERGENT_SCRIPT)

    for tag in [PICO_TAG, MODEL_TAG]:
        print(f"\n{'='*60}")
        print(f"MODEL: {tag}")
        print(f"{'='*60}")
        questions_args = " ".join(f'"{q}"' for q in EMERGENT_QUESTIONS)
        _run(
            f"cd /root/nanochat && "
            f"uv run python {script_path} {tag} {questions_args}"
        )
    print("\nDone. Copy the outputs above into the Part 4 writeup.")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Full Part 4 pipeline:
        0. Download 370 FineWeb-EDU shards      (CPU,    ~20 min,  ~$0.50)
        1. Train BPE tokenizer                  (1×H100, ~2 min,   ~$0.12)
        2. Pretrain nanochat d=26 + DiffAttn    (8×H100, ~3 hrs,   ~$84.00)
        3. SFT fine-tuning                      (8×H100, ~30 min,  ~$14.00)
        4. Eval (bpb + CORE)                    (4×H100, ~30 min,  ~$7.00)

    Estimated total: ~$105-110 at H100 on-demand pricing (~$3.50/GPU/hr).
    """
    w = 64
    print("\n" + "=" * w)
    print("Part 4: Final Nanochat — d=26 + Differential Attention")
    print(f"  Model tag : {MODEL_TAG}")
    print(f"  GPU       : {GPU_TRAIN}  |  Shards: {NUM_SHARDS}")
    print(f"  WandB     : {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[0/7] Downloading FineWeb-EDU shards...")
    stage_data.remote(num_shards=NUM_SHARDS)

    print("[1/7] Training tokenizer...")
    stage_tokenizer.remote()

    print("[2/7] Pretraining picochat baseline d=12...")
    stage_pretrain_pico.remote()

    print("[3/7] SFT fine-tuning picochat...")
    stage_sft_pico.remote()

    print("[4/7] Pretraining nanochat d=26 + DiffAttn...")
    stage_pretrain.remote()

    print("[5/7] SFT fine-tuning nanochat...")
    stage_sft.remote()

    print("[6/7] Evaluating nanochat (bpb + CORE)...")
    stage_eval.remote()

    print("[7/7] Emergent abilities questions...")
    stage_emergent_abilities.remote()

    print("\n" + "=" * w)
    print("Part 4 complete!")
    print(f"  Checkpoints: Modal Volume 'nanochat-vol' under {MODEL_TAG}/")
    print(f"  W&B: https://wandb.ai/your-username/{WANDB_PROJECT}")
    print("=" * w + "\n")
