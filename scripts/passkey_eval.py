"""
Passkey Retrieval Evaluation
=============================

Tests whether a model can retrieve a 5-digit passkey placed at varying
distances from the query. Models with limited attention windows (e.g.,
seq_len=512) will fail when the passkey is beyond their window.

Usage:
    python -m scripts.passkey_eval \
        --model-tags pico_ctx512 pico_ctx2048 \
        --filler-lengths 256 512 768 1024 1536 \
        --trials 50 --seed 42 \
        --output /path/to/passkey_results.json
"""

import argparse
import json
import os
import sys

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.tokenizer import get_tokenizer


def run_passkey_eval(
    model_tags: list[str],
    filler_lengths: list[int],
    trials: int,
    seed: int,
    output_path: str | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
    checkpoints_base = os.path.join(base_dir, "base_checkpoints")

    # Pre-tokenize filler text
    filler_text = (
        "This is some filler text that is used to pad the context window. "
        "The quick brown fox jumps over the lazy dog. "
        "In a hole in the ground there lived a hobbit. "
        "It was the best of times, it was the worst of times. "
        "To be or not to be, that is the question. "
        "All that glitters is not gold. "
        "The only thing we have to fear is fear itself. "
    )
    filler_tokens = tokenizer.encode(filler_text * 50)  # ~2000+ tokens

    rng = torch.Generator()
    rng.manual_seed(seed)

    all_results = {}

    for tag in model_tags:
        print(f"\n{'='*60}")
        print(f"Passkey Retrieval Eval: {tag}")
        print(f"{'='*60}")

        checkpoint_dir = os.path.join(checkpoints_base, tag)
        last_step = find_last_step(checkpoint_dir)
        model, _, meta = build_model(checkpoint_dir, last_step, device, phase="eval")

        tag_results = {}
        for filler_len in filler_lengths:
            correct = 0
            for trial in range(trials):
                passkey = str(10000 + torch.randint(0, 90000, (1,), generator=rng).item())

                bos = tokenizer.get_bos_token_id()
                prefix_tokens = tokenizer.encode(f" The passkey is {passkey}.")
                query_tokens = tokenizer.encode(" What is the passkey? The passkey is")
                target_tokens = tokenizer.encode(f" {passkey[0]}")
                target_token = target_tokens[0]

                filler = filler_tokens[:filler_len]
                input_ids = [bos] + prefix_tokens + filler + query_tokens
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_tensor)

                pred_token = logits[0, -1, :].argmax().item()
                if pred_token == target_token:
                    correct += 1

            accuracy = correct / trials
            tag_results[filler_len] = accuracy
            print(f"  filler_len={filler_len:5d} tokens | accuracy={accuracy:.2%} ({correct}/{trials})")

        all_results[tag] = tag_results
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("  PASSKEY RETRIEVAL SUMMARY")
    print(f"{'='*60}")
    header = f"{'Filler Length':>15}"
    for tag in model_tags:
        header += f"  {tag:>15}"
    print(header)
    print("-" * 60)
    for filler_len in filler_lengths:
        row = f"{filler_len:>15}"
        for tag in model_tags:
            acc = all_results[tag][filler_len]
            row += f"  {acc:>14.1%}"
        print(row)
    print("=" * 60)

    # Save results
    if output_path:
        serializable = {}
        for tag, res in all_results.items():
            serializable[tag] = {str(k): v for k, v in res.items()}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nPasskey results saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Passkey retrieval evaluation")
    parser.add_argument("--model-tags", nargs="+", required=True, help="Model checkpoint tags to evaluate")
    parser.add_argument("--filler-lengths", nargs="+", type=int, default=[256, 512, 768, 1024, 1536])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    run_passkey_eval(
        model_tags=args.model_tags,
        filler_lengths=args.filler_lengths,
        trials=args.trials,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
