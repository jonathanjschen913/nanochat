"""
Collect GSM8K completions from a trained RL model for error analysis.
Supports DDP: each rank handles a subset of problems, rank 0 merges all.

Usage:
    python -m scripts.collect_completions --model-tag=sft_combo --output=part3_completions.jsonl
"""
import argparse
import json
import os
import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir, autodetect_device_type, compute_init, compute_cleanup
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--source", type=str, default="rl", choices=["sft", "rl"])
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-problems", type=int, default=None)
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.bfloat16 if device_type == "cuda" else torch.float32

model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
model.eval()
model = model.to(ptdtype)
engine = Engine(model, tokenizer)

task = GSM8K(subset="main", split="test")
n = min(args.max_problems, len(task)) if args.max_problems else len(task)

out_path = args.output or os.path.join(get_base_dir(), f"{args.model_tag}_completions.jsonl")
rank_out_path = out_path.replace(".jsonl", f"_rank{ddp_rank}.jsonl")

if ddp_rank == 0:
    print(f"Collecting {n} completions across {ddp_world_size} GPUs -> {out_path}")

records = []
for idx in range(ddp_rank, n, ddp_world_size):
    conversation = task[idx]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    with torch.no_grad():
        seqs, _ = engine.generate_batch(
            tokens, num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    completion = tokenizer.decode(seqs[0][prefix_length:])
    is_correct = bool(task.evaluate(conversation, completion))
    gold = extract_answer(conversation["messages"][-1]["content"][-1]["text"])
    question = conversation["messages"][0]["content"]

    records.append({
        "idx": idx,
        "question": question,
        "gold_answer": gold,
        "completion": completion,
        "is_correct": is_correct,
    })

    if len(records) % 10 == 0:
        print(f"  Rank {ddp_rank} | {len(records)} done")

# Write per-rank file
with open(rank_out_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

# Rank 0 merges all rank files after barrier
if ddp:
    dist.barrier()

if ddp_rank == 0:
    all_records = []
    for rank in range(ddp_world_size):
        rpath = out_path.replace(".jsonl", f"_rank{rank}.jsonl")
        with open(rpath) as f:
            all_records.extend(json.loads(l) for l in f)
        os.remove(rpath)
    # Sort by idx to maintain original order
    all_records.sort(key=lambda r: r["idx"])
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    num_correct = sum(r["is_correct"] for r in all_records)
    total = len(all_records)
    print(f"\nDone. Saved {total} completions to {out_path}")
    print(f"pass@1 greedy: {num_correct}/{total} = {num_correct/total:.1%}")
    print(f"Karpathy after RL: 7.58% | Our SFT baseline: 20.0%")

compute_cleanup()
