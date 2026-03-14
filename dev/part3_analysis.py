"""
Part 3 Error Analysis — GSM8K Completions
==========================================

Reads dev/part3_completions.jsonl (downloaded from Modal after stage_collect_completions)
and produces:
  1. Overall accuracy
  2. Error category breakdown (bar chart)
  3. Accuracy vs question length (scatter/box plot)
  4. Accuracy vs answer magnitude (box plot)
  5. Problem cluster accuracy (bar chart)
  6. Sample correct and incorrect completions printed to stdout

Usage:
    python dev/part3_analysis.py
    python dev/part3_analysis.py --input dev/part3_completions.jsonl --output dev/part3_plots/
"""

import argparse
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="dev/part3_completions.jsonl")
parser.add_argument("--output", default="dev/part3_plots")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

records = []
with open(args.input) as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")
total = len(records)
correct = sum(r["is_correct"] for r in records)
print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}")

# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def categorize(record):
    completion = record["completion"]
    gold = record["gold_answer"]
    is_correct = record["is_correct"]

    if is_correct:
        return "Correct"

    # Truncated: completion fills token budget (heuristic: ends without #### and is long)
    if len(completion.split()) > 200 and not GSM_RE.search(completion):
        return "Truncated"

    # Missing answer: no #### marker at all
    pred_match = GSM_RE.search(completion)
    if pred_match is None:
        return "Missing answer"

    # Has an answer — compare numerically
    pred_str = pred_match.group(1).replace(",", "")
    gold_str = (gold or "").replace(",", "")
    try:
        pred_val = float(pred_str)
        gold_val = float(gold_str)
    except ValueError:
        return "Wrong answer (non-numeric)"

    diff = abs(pred_val - gold_val)
    ratio = abs(pred_val / gold_val) if gold_val != 0 else float("inf")

    # Unit/scale error: answer is a clean multiple or fraction of gold
    if gold_val != 0 and diff > 0:
        for factor in [60, 100, 12, 24, 7, 1000, 0.5, 2]:
            if abs(pred_val - gold_val * factor) < 0.01 or abs(pred_val * factor - gold_val) < 0.01:
                return "Unit/scale error"

    # Small arithmetic error: within 10% of gold
    if gold_val != 0 and abs(ratio - 1.0) < 0.10:
        return "Arithmetic error (small)"

    return "Wrong setup / large error"


for r in records:
    r["error_category"] = categorize(r)

category_counts = Counter(r["error_category"] for r in records)
print("\nError categories:")
for cat, count in category_counts.most_common():
    print(f"  {cat}: {count} ({count/total:.1%})")

# Plot 1: Error category bar chart
fig, ax = plt.subplots(figsize=(9, 5))
cats = [c for c, _ in category_counts.most_common()]
counts = [category_counts[c] for c in cats]
colors = ["#2ecc71" if c == "Correct" else "#e74c3c" for c in cats]
bars = ax.bar(cats, counts, color=colors, edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=3)
ax.set_ylabel("Count")
ax.set_title("GSM8K Error Categories (RL model, test set)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(args.output, "error_categories.png"), dpi=150)
plt.close()
print(f"\nSaved: {args.output}/error_categories.png")

# ---------------------------------------------------------------------------
# Accuracy vs question length
# ---------------------------------------------------------------------------

for r in records:
    r["question_words"] = len(r["question"].split())

# Bin by question length
bins = [0, 30, 50, 70, 90, 200]
bin_labels = ["<30", "30-50", "50-70", "70-90", "90+"]
bin_correct = [0] * len(bin_labels)
bin_total  = [0] * len(bin_labels)

for r in records:
    qw = r["question_words"]
    for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
        if lo <= qw < hi:
            bin_total[i] += 1
            bin_correct[i] += r["is_correct"]
            break

bin_acc = [c / t if t > 0 else 0 for c, t in zip(bin_correct, bin_total)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(bin_labels, bin_acc, color="#3498db", edgecolor="white")
for i, (acc, tot) in enumerate(zip(bin_acc, bin_total)):
    ax.text(i, acc + 0.005, f"{acc:.0%}\n(n={tot})", ha="center", fontsize=9)
ax.set_xlabel("Question length (words)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy vs Question Length")
plt.tight_layout()
plt.savefig(os.path.join(args.output, "accuracy_vs_length.png"), dpi=150)
plt.close()
print(f"Saved: {args.output}/accuracy_vs_length.png")

# ---------------------------------------------------------------------------
# Accuracy vs answer magnitude
# ---------------------------------------------------------------------------

buckets = {"1-10": [], "11-100": [], "101-1000": [], "1000+": []}
for r in records:
    try:
        val = abs(float((r["gold_answer"] or "0").replace(",", "")))
    except ValueError:
        continue
    if val <= 10:
        buckets["1-10"].append(r["is_correct"])
    elif val <= 100:
        buckets["11-100"].append(r["is_correct"])
    elif val <= 1000:
        buckets["101-1000"].append(r["is_correct"])
    else:
        buckets["1000+"].append(r["is_correct"])

fig, ax = plt.subplots(figsize=(8, 4))
bucket_labels = list(buckets.keys())
bucket_acc = [np.mean(v) if v else 0 for v in buckets.values()]
bucket_n   = [len(v) for v in buckets.values()]
bars = ax.bar(bucket_labels, bucket_acc, color="#9b59b6", edgecolor="white")
for i, (acc, n) in enumerate(zip(bucket_acc, bucket_n)):
    ax.text(i, acc + 0.005, f"{acc:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_xlabel("Gold answer magnitude")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy vs Answer Magnitude")
plt.tight_layout()
plt.savefig(os.path.join(args.output, "accuracy_vs_magnitude.png"), dpi=150)
plt.close()
print(f"Saved: {args.output}/accuracy_vs_magnitude.png")

# ---------------------------------------------------------------------------
# Problem cluster accuracy (keyword-based)
# ---------------------------------------------------------------------------

CLUSTERS = {
    "Rate/time/speed": [r"hour", r"minute", r"speed", r"mph", r"per hour", r"rate"],
    "Money/cost":      [r"\$", r"dollar", r"cent", r"cost", r"price", r"earn", r"pay"],
    "Percentage":      [r"percent", r"%", r"discount", r"tax", r"markup"],
    "Counting/groups": [r"each", r"total", r"group", r"class", r"student", r"people"],
    "Fractions":       [r"fraction", r"half", r"third", r"quarter", r"ratio"],
    "Geometry/area":   [r"area", r"perimeter", r"square", r"rectangle", r"length", r"width"],
}

cluster_correct = Counter()
cluster_total   = Counter()

for r in records:
    q = r["question"].lower()
    matched = False
    for cluster, patterns in CLUSTERS.items():
        if any(re.search(p, q) for p in patterns):
            cluster_total[cluster] += 1
            cluster_correct[cluster] += r["is_correct"]
            matched = True
            break
    if not matched:
        cluster_total["Other"] += 1
        cluster_correct["Other"] += r["is_correct"]

cluster_labels = list(cluster_total.keys())
cluster_acc = [cluster_correct[c] / cluster_total[c] for c in cluster_labels]
cluster_n   = [cluster_total[c] for c in cluster_labels]

# Sort by accuracy
order = sorted(range(len(cluster_labels)), key=lambda i: cluster_acc[i], reverse=True)
cluster_labels = [cluster_labels[i] for i in order]
cluster_acc    = [cluster_acc[i] for i in order]
cluster_n      = [cluster_n[i] for i in order]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(cluster_labels, cluster_acc, color="#e67e22", edgecolor="white")
for i, (acc, n) in enumerate(zip(cluster_acc, cluster_n)):
    ax.text(i, acc + 0.005, f"{acc:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy by Problem Category")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(args.output, "accuracy_by_cluster.png"), dpi=150)
plt.close()
print(f"Saved: {args.output}/accuracy_by_cluster.png")

# ---------------------------------------------------------------------------
# Print sample completions
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("SAMPLE CORRECT COMPLETIONS (3)")
print("="*60)
correct_samples = [r for r in records if r["is_correct"]][:3]
for r in correct_samples:
    print(f"\n[Q] {r['question'][:120]}...")
    print(f"[Gold] {r['gold_answer']}")
    print(f"[Model] {r['completion'][:300]}...")

print("\n" + "="*60)
print("SAMPLE INCORRECT COMPLETIONS BY CATEGORY")
print("="*60)
shown = set()
for r in records:
    cat = r["error_category"]
    if cat != "Correct" and cat not in shown:
        shown.add(cat)
        print(f"\n[Category: {cat}]")
        print(f"[Q] {r['question'][:120]}...")
        print(f"[Gold] {r['gold_answer']}")
        print(f"[Model] {r['completion'][:300]}...")

print(f"\nAll plots saved to: {args.output}/")
print("Include these in your A4 Part 3 writeup.")
