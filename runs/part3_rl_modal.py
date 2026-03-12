"""
Part 3: RL on GSM8K — Modal Training Script
============================================

Replicates Karpathy's GSM8K RL run starting from our best Part 2 SFT checkpoint
(sft_combo: baseline + MetaMathQA + DART-Math-Hard, 20% GSM8K before RL).

Two runs:
  Run 1 (rl_baseline): RL starting from sft_combo — our primary run
  Run 2 (rl_collect):  Same run but saves full completions for error analysis

Usage
-----
Full pipeline:
    modal run runs/part3_rl_modal.py

Individual stages:
    modal run runs/part3_rl_modal.py::stage_rl_baseline
    modal run runs/part3_rl_modal.py::stage_collect_completions
    modal run runs/part3_rl_modal.py::stage_eval

Cost estimate (8×H100 at ~$28/hr, eval on 4×H100 at ~$14/hr)
--------------------------------------------------------------
    RL training (467 steps + 7 evals)  : ~70 min   ~$33
    Completion collection (test set)   : ~20 min   ~$5
    Eval (pass@1, pass@8)              : ~20 min   ~$5
    Total                                          ~$52
"""

import json
import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# SFT checkpoint to start RL from (best from Part 2)
SFT_TAG = "sft_combo"

# RL checkpoint output tags
RL_TAG = "rl_baseline"

# GPU config
GPU_TRAIN = "H100:8"
GPU_EVAL  = "H100:1"

_N_TRAIN_GPUS = int(GPU_TRAIN.split(":")[1])
_N_EVAL_GPUS  = int(GPU_EVAL.split(":")[1])

# RL hyperparameters — match Karpathy's defaults
DEVICE_BATCH_SIZE  = 8   # d=26 OOM at 16 during generation
EXAMPLES_PER_STEP  = 16
NUM_SAMPLES        = 16
MAX_NEW_TOKENS     = 256
TEMPERATURE        = 1.0
TOP_K              = 50
NUM_EPOCHS         = 1  # = 467 steps; at ~1 min/step = ~8 hrs — consider killing at ~200 steps if needed

# W&B
# NOTE: chat_rl.py hardcodes project="nanochat-rl" — only the run *name* uses this value
WANDB_PROJECT = "nanochat-part3"

# Timeouts
RL_TIMEOUT_SEC   = 60 * 60 * 24  # 24 hours
EVAL_TIMEOUT_SEC = 60 * 60 * 1   # 1 hour

# Volume / paths
VOLUME_MOUNT  = "/vol"
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
# STAGE: RL TRAINING (Run 1 — baseline replication)
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
    Run 1: RL on GSM8K starting from sft_combo.
    Replicates Karpathy's default chat_rl.py settings.
    Logs reward curve and pass@k to W&B project nanochat-part3.
    """
    _setup_cache()

    # Verify the Part 2 SFT checkpoint exists before spending GPU time
    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", SFT_TAG)
    if not os.path.isdir(sft_dir):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_dir}\n"
            f"Run Part 2 first: modal run runs/part2_sft_modal.py::stage_sft_combo"
        )
    print(f"Found SFT checkpoint: {sft_dir}")

    args = [
        f"--run={WANDB_PROJECT}_{RL_TAG}",
        f"--model-tag={SFT_TAG}",        # load from chatsft_checkpoints/sft_combo
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--examples-per-step={EXAMPLES_PER_STEP}",
        f"--num-samples={NUM_SAMPLES}",
        f"--max-new-tokens={MAX_NEW_TOKENS}",
        f"--temperature={TEMPERATURE}",
        f"--top-k={TOP_K}",
        f"--num-epochs={NUM_EPOCHS}",
        "--eval-every=999",   # effectively disabled (step 0 skipped via chat_rl.py fix); run stage_eval separately
        "--eval-examples=400",
        "--save-every=60",
    ]
    _torchrun("scripts.chat_rl", args, nproc=_N_TRAIN_GPUS)
    volume.commit()
    print(f"RL training complete. Checkpoint saved to chatrl_checkpoints/{SFT_TAG}/")


# =============================================================================
# STAGE: COLLECT COMPLETIONS (for Part 3 error analysis)
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
    Run the RL-trained model on the full GSM8K test set and save completions
    to BASE_DIR/part3_completions.jsonl for error analysis.

    Each line: {"idx": int, "question": str, "gold_answer": str,
                "completion": str, "is_correct": bool}
    """
    _setup_cache()

    import sys
    sys.path.insert(0, "/root/nanochat")

    import torch
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    from tasks.gsm8k import GSM8K, extract_answer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the RL checkpoint
    model, tokenizer, _ = load_model(
        "rl", device, phase="eval", model_tag=SFT_TAG
    )
    model.eval()

    engine = Engine(model, tokenizer)

    task = GSM8K(subset="main", split="test")
    out_path = os.path.join(BASE_DIR, "part3_completions.jsonl")

    print(f"Collecting completions for {len(task)} test examples -> {out_path}")

    with open(out_path, "w") as f:
        for idx in range(len(task)):
            conversation = task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            with torch.no_grad():
                seqs, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=256,
                    temperature=0.0, top_k=50,
                )

            generated_tokens = seqs[0][prefix_length:]
            completion = tokenizer.decode(generated_tokens)
            is_correct = bool(task.evaluate(conversation, completion))

            # Extract gold answer from conversation
            last_part = conversation["messages"][-1]["content"][-1]["text"]
            gold = extract_answer(last_part)
            question = conversation["messages"][0]["content"]

            record = {
                "idx": idx,
                "question": question,
                "gold_answer": gold,
                "completion": completion,
                "is_correct": is_correct,
            }
            f.write(json.dumps(record) + "\n")

            if idx % 50 == 0:
                print(f"  [{idx}/{len(task)}] is_correct={is_correct}")

    volume.commit()
    print(f"Saved {len(task)} completions to {out_path}")


# =============================================================================
# STAGE: FINAL EVAL — GSM8K pass@1 greedy (matches Karpathy's reported metric)
# =============================================================================

# Karpathy's published numbers from https://github.com/karpathy/nanochat/discussions/1
KARPATHY_RESULTS = {
    "after_midtraining": 0.0250,
    "after_sft":         0.0455,
    "after_rl":          0.0758,  # <-- what we compare our RL run against
}

# Our Part 2 SFT result (from a4_part2_writeup.md, n=50 eval)
OUR_SFT_RESULT = 0.20  # sft_combo, 20% GSM8K


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_EVAL,
    timeout=EVAL_TIMEOUT_SEC,
)
def stage_eval() -> dict:
    """
    Evaluate the RL checkpoint on the full GSM8K test set (1319 problems).
    - pass@1 greedy (temperature=0): directly comparable to Karpathy's 7.58%
    - pass@1 sampled (temperature=1): matches the RL training eval metric
    - pass@8 sampled: upper bound on model capability

    Uses KV-cached generation (diff_attn KV cache now supported in gpt.py).
    Returns result dict so the local entrypoint can print the comparison table.
    """
    _setup_cache()

    import sys
    sys.path.insert(0, "/root/nanochat")

    import torch
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    from tasks.gsm8k import GSM8K

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load RL checkpoint (saved to chatrl_checkpoints/sft_combo by chat_rl.py)
    model, tokenizer, _ = load_model("rl", device, phase="eval", model_tag=SFT_TAG)
    model.eval()

    engine = Engine(model, tokenizer)
    task = GSM8K(subset="main", split="test")

    def eval_pass_at_k(temperature, num_samples, max_examples=None):
        """Returns pass@k accuracy over the test set."""
        n = min(max_examples, len(task)) if max_examples else len(task)
        pass_k = 0
        for idx in range(n):
            conversation = task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            with torch.no_grad():
                seqs, _ = engine.generate_batch(
                    tokens, num_samples=num_samples,
                    max_tokens=256, temperature=temperature, top_k=50,
                )
            completions = [tokenizer.decode(s[prefix_length:]) for s in seqs]
            if any(task.evaluate(conversation, c) for c in completions):
                pass_k += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{n}] running pass@{num_samples}={pass_k/(idx+1):.3f}")
        return pass_k / n

    print("\n--- pass@1 greedy (temperature=0) — comparable to Karpathy's 7.58% ---")
    pass1_greedy = eval_pass_at_k(temperature=0.0, num_samples=1)

    print("\n--- pass@1 sampled (temperature=1) — matches RL training eval ---")
    pass1_sampled = eval_pass_at_k(temperature=1.0, num_samples=1)

    print("\n--- pass@8 sampled — upper bound on capability ---")
    pass8 = eval_pass_at_k(temperature=1.0, num_samples=8)

    results = {
        "pass@1_greedy":  pass1_greedy,
        "pass@1_sampled": pass1_sampled,
        "pass@8":         pass8,
    }

    # Save results to volume for local download
    out_path = os.path.join(BASE_DIR, "part3_eval_results.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    print("\n" + "="*60)
    print("COMPARISON TO KARPATHY (discussion #1)")
    print("="*60)
    print(f"{'Metric':<30} {'Karpathy':>12} {'Ours':>12}")
    print("-"*55)
    print(f"{'After SFT (pass@1 greedy)':<30} {KARPATHY_RESULTS['after_sft']:>11.1%} {OUR_SFT_RESULT:>11.1%}")
    print(f"{'After RL  (pass@1 greedy)':<30} {KARPATHY_RESULTS['after_rl']:>11.1%} {pass1_greedy:>11.1%}")
    print(f"{'After RL  (pass@1 sampled)':<30} {'N/A':>12} {pass1_sampled:>11.1%}")
    print(f"{'After RL  (pass@8 sampled)':<30} {'N/A':>12} {pass8:>11.1%}")
    print("="*60)

    return results


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
    path = os.path.join(BASE_DIR, "part3_completions.jsonl")
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
    print("Part 3: RL on GSM8K — Replication Run")
    print(f"  SFT starting point : {SFT_TAG} (20% GSM8K from Part 2)")
    print(f"  RL output tag      : {RL_TAG}")
    print(f"  GPU                : {GPU_TRAIN}")
    print(f"  WandB              : {WANDB_PROJECT}")
    print("=" * w + "\n")

    print("[1/4] RL training...")
    stage_rl_baseline.remote()

    print("[2/4] Collecting completions for error analysis...")
    stage_collect_completions.remote()

    print("[3/4] Final eval (pass@1 greedy, pass@1 sampled, pass@8)...")
    eval_results = stage_eval.remote()

    print("[4/4] Downloading completions locally...")
    data = stage_read_completions.remote()
    local_path = "dev/part3_completions.jsonl"
    with open(local_path, "wb") as f:
        f.write(data)
    print(f"Completions saved locally to: {local_path}")

    print("\n" + "=" * w)
    print("FINAL COMPARISON — Our Run vs Karpathy (discussion #1)")
    print("=" * w)
    print(f"  Karpathy after SFT (pass@1 greedy) : {KARPATHY_RESULTS['after_sft']:.1%}")
    print(f"  Ours     after SFT (pass@1 greedy) : {OUR_SFT_RESULT:.1%}  (+{OUR_SFT_RESULT - KARPATHY_RESULTS['after_sft']:.1%} vs Karpathy SFT)")
    print(f"  Karpathy after RL  (pass@1 greedy) : {KARPATHY_RESULTS['after_rl']:.1%}")
    if eval_results:
        our_rl = eval_results['pass@1_greedy']
        print(f"  Ours     after RL  (pass@1 greedy) : {our_rl:.1%}  (+{our_rl - KARPATHY_RESULTS['after_rl']:.1%} vs Karpathy RL)")
        print(f"  Ours     after RL  (pass@1 sampled): {eval_results['pass@1_sampled']:.1%}")
        print(f"  Ours     after RL  (pass@8 sampled): {eval_results['pass@8']:.1%}")
    print("=" * w)
    print(f"\nCompletions : {local_path}")
    print("Run analysis: python dev/part3_analysis.py")
    print("W&B project : nanochat-part3  (reward + pass@k curves during training)")
    print("=" * w + "\n")
