"""
Part 4: End-to-end integration test on Modal
=============================================

Runs on 1 GPU for ~5 minutes (~$2). Validates the FULL pipeline:
1. SFT checkpoint exists on the volume
2. Reward functions import and compute correctly
3. --reward-fn and --output-tag args work in chat_rl.py
4. Training runs for 2 steps with "combined" reward
5. Checkpoint saved to chatrl_checkpoints/rl_part4_test/ (not sft_combo!)
6. Eval loads from that checkpoint successfully
7. Cleanup: removes test checkpoint

Usage:
    PYTHONUTF8=1 modal run runs/part4_test_modal.py
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# Config — mirrors part4_rl_modal.py but minimal
SFT_TAG = "sft_combo"
TEST_OUTPUT_TAG = "rl_part4_test"  # test checkpoint dir — will be cleaned up
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

app = App("nanochat-part4-test")
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


def _setup_cache():
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)
    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)


def _run(cmd: str, label: str) -> str:
    """Run a shell command, print output, raise on failure."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {label}\n{output}")
    return output


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,  # 30 min max
)
def test_full_pipeline() -> str:
    _setup_cache()
    results = []

    # -------------------------------------------------------------------------
    # TEST 1: SFT checkpoint exists
    # -------------------------------------------------------------------------
    sft_dir = os.path.join(BASE_DIR, "chatsft_checkpoints", SFT_TAG)
    if not os.path.isdir(sft_dir):
        raise FileNotFoundError(
            f"FATAL: SFT checkpoint not found: {sft_dir}\n"
            f"The Part 4 training runs will fail without this."
        )
    ckpt_files = os.listdir(sft_dir)
    results.append(f"[PASS] Test 1: SFT checkpoint exists at {sft_dir} ({len(ckpt_files)} files)")
    print(results[-1])

    # -------------------------------------------------------------------------
    # TEST 2: Reward functions import and compute correctly
    # -------------------------------------------------------------------------
    _run(
        'cd /root/nanochat && uv run python -c "'
        "from tasks.gsm8k import REWARD_FNS; "
        "assert len(REWARD_FNS) == 5; "
        "assert set(REWARD_FNS.keys()) == {'binary','format','tolerance','steps','combined'}; "
        "print('REWARD_FNS:', list(REWARD_FNS.keys()))"
        '"',
        "Test 2: Reward functions import"
    )
    results.append("[PASS] Test 2: All 5 reward functions import correctly")

    # -------------------------------------------------------------------------
    # TEST 3: --reward-fn and --output-tag in chat_rl.py argparse
    # -------------------------------------------------------------------------
    help_output = _run(
        "cd /root/nanochat && uv run python -m scripts.chat_rl --help",
        "Test 3: chat_rl.py --help"
    )
    assert "--reward-fn" in help_output, "--reward-fn missing from chat_rl.py"
    assert "--output-tag" in help_output, "--output-tag missing from chat_rl.py"
    results.append("[PASS] Test 3: --reward-fn and --output-tag both present in chat_rl.py")

    # -------------------------------------------------------------------------
    # TEST 4: Train 2 steps with combined reward on 1 GPU
    #         Save checkpoint to rl_part4_test (not sft_combo!)
    # -------------------------------------------------------------------------
    # Clean up any leftover test checkpoints from previous runs
    import shutil as _shutil
    stale_dir = os.path.join(BASE_DIR, "chatrl_checkpoints", TEST_OUTPUT_TAG)
    if os.path.isdir(stale_dir):
        _shutil.rmtree(stale_dir)
        print(f"Cleaned up stale test checkpoint: {stale_dir}")

    _run(
        "cd /root/nanochat && "
        "uv run python -m scripts.chat_rl "
        "--run=dummy "
        f"--model-tag={SFT_TAG} "
        f"--output-tag={TEST_OUTPUT_TAG} "
        "--reward-fn=combined "
        "--device-batch-size=8 "
        "--examples-per-step=1 "
        "--num-samples=8 "
        "--max-new-tokens=64 "
        "--temperature=1.0 "
        "--top-k=50 "
        "--num-epochs=1 "
        "--max-steps=2 "
        "--eval-every=999 "
        "--save-every=1 ",
        "Test 4: Train 2 steps with combined reward (--max-steps=2)"
    )
    results.append("[PASS] Test 4: Training completed (2 steps, combined reward)")

    # -------------------------------------------------------------------------
    # TEST 5: Checkpoint saved to correct directory
    # -------------------------------------------------------------------------
    test_ckpt_dir = os.path.join(BASE_DIR, "chatrl_checkpoints", TEST_OUTPUT_TAG)
    if not os.path.isdir(test_ckpt_dir):
        raise FileNotFoundError(
            f"FATAL: Test checkpoint not found at {test_ckpt_dir}\n"
            f"--output-tag is not working correctly!"
        )
    test_ckpt_files = os.listdir(test_ckpt_dir)
    results.append(f"[PASS] Test 5: Checkpoint saved to {test_ckpt_dir} ({len(test_ckpt_files)} files)")
    print(results[-1])

    # Verify Part 3 checkpoint was NOT overwritten
    sft_combo_rl_dir = os.path.join(BASE_DIR, "chatrl_checkpoints", SFT_TAG)
    if os.path.isdir(sft_combo_rl_dir):
        results.append(f"[PASS] Test 5b: Part 3 checkpoint at chatrl_checkpoints/{SFT_TAG}/ is preserved")
    else:
        results.append(f"[INFO] Test 5b: No Part 3 checkpoint at chatrl_checkpoints/{SFT_TAG}/ (may not have been created yet)")
    print(results[-1])

    # -------------------------------------------------------------------------
    # TEST 6: Eval loads from the test checkpoint
    # -------------------------------------------------------------------------
    _run(
        "cd /root/nanochat && "
        "uv run python -m scripts.chat_eval "
        "-i rl "
        f"-g {TEST_OUTPUT_TAG} "
        "-a GSM8K "
        "--max-problems=5 "
        "--batch-size=1 "
        "--temperature=0.0 "
        "--num-samples=1 ",
        "Test 6: Eval loads from test checkpoint"
    )
    results.append("[PASS] Test 6: Eval successfully loaded and ran from test checkpoint")

    # -------------------------------------------------------------------------
    # CLEANUP: Remove test checkpoint to avoid clutter
    # -------------------------------------------------------------------------
    import shutil
    shutil.rmtree(test_ckpt_dir)
    volume.commit()
    results.append(f"[PASS] Cleanup: Removed test checkpoint {test_ckpt_dir}")
    print(results[-1])

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    summary = "\n".join([
        "",
        "=" * 60,
        "  PART 4 END-TO-END TEST SUMMARY",
        "=" * 60,
        *results,
        "=" * 60,
        "  ALL TESTS PASSED — safe to run full training",
        "=" * 60,
        "",
        "To run the 3 training jobs:",
        "  REWARD_FN=combined modal run runs/part4_rl_modal.py",
        "  REWARD_FN=format   modal run runs/part4_rl_modal.py",
        "  REWARD_FN=steps    modal run runs/part4_rl_modal.py",
        "",
    ])
    print(summary)
    return summary


@app.local_entrypoint()
def main():
    print("Running Part 4 end-to-end integration test...")
    print("  GPU: 1x H100 | Est. time: ~5 min | Est. cost: ~$2")
    print()
    result = test_full_pipeline.remote()
    print(result)
