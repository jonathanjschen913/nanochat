"""Generate all Part 4 plots for the writeup."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
OUT_DIR = "dev/part4_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# DATA
# ============================================================================

models = ["Binary\n(Part 3)", "Format", "Steps", "Tolerance", "Combined"]
short_names = ["binary", "format", "steps", "tolerance", "combined"]

pass1 = [25.5, 25.55, 25.17, 24.64, 26.38]
pass8 = [35.75, 38.50, 39.50, 39.75, 37.00]

# Error categories (from analysis)
correct =      [337, 337, 332, 325, 348]
missing_ans =  [44,  10,  21,  14,  16]
unit_scale =   [25,  21,  20,  14,  24]
arith_small =  [371, 374, 351, 358, 349]
wrong_setup =  [542, 577, 595, 608, 582]

avg_chars = [470, 818, 808, 796, 799]
avg_lines = [6.1, 12.5, 16.5, 12.7, 14.8]
unique_solved = [37, 44, 35, 34, 42]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ============================================================================
# PLOT 1: pass@1 and pass@8 grouped bar chart
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, pass1, w, label="pass@1 (greedy)", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + w/2, pass8, w, label="pass@8 (sampled)", color="#DD8452", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
ax.set_title("Pass@1 and Pass@8 Across Reward Functions", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 48)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_pass_at_k.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_pass_at_k.png")

# ============================================================================
# PLOT 2: Error category stacked bar chart
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
w = 0.6

# Convert to percentages
total = 1319
c_pct = [v/total*100 for v in correct]
m_pct = [v/total*100 for v in missing_ans]
u_pct = [v/total*100 for v in unit_scale]
a_pct = [v/total*100 for v in arith_small]
ws_pct = [v/total*100 for v in wrong_setup]

b1 = ax.bar(x, c_pct, w, label="Correct", color="#55A868")
b2 = ax.bar(x, m_pct, w, bottom=c_pct, label="Missing ####", color="#C44E52")
bottom2 = [c + m for c, m in zip(c_pct, m_pct)]
b3 = ax.bar(x, u_pct, w, bottom=bottom2, label="Unit/scale error", color="#DD8452")
bottom3 = [b + u for b, u in zip(bottom2, u_pct)]
b4 = ax.bar(x, a_pct, w, bottom=bottom3, label="Arithmetic (small)", color="#8172B3")
bottom4 = [b + a for b, a in zip(bottom3, a_pct)]
b5 = ax.bar(x, ws_pct, w, bottom=bottom4, label="Wrong setup / large", color="#4C72B0")

ax.set_ylabel("% of 1319 Test Problems", fontsize=12)
ax.set_title("Error Category Distribution by Reward Function", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(0, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_error_categories.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_error_categories.png")

# ============================================================================
# PLOT 3: Missing answer (####) reduction — focused bar chart
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models, missing_ans, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, missing_ans):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel("Count (out of 1319)", fontsize=12)
ax.set_title("Missing Answer (#### absent) Errors by Reward", fontsize=14)
ax.set_ylim(0, 55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=44, color='gray', linestyle='--', alpha=0.5, label='Binary baseline')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_missing_answer.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_missing_answer.png")

# ============================================================================
# PLOT 4: Response length comparison
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars1 = ax1.bar(models, avg_chars, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars1, avg_chars):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_ylabel("Average Characters", fontsize=12)
ax1.set_title("Average Response Length (chars)", fontsize=13)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, 950)

bars2 = ax2.bar(models, avg_lines, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars2, avg_lines):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel("Average Lines", fontsize=12)
ax2.set_title("Average Response Length (lines)", fontsize=13)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, 20)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_response_length.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_response_length.png")

# ============================================================================
# PLOT 5: Unique problems solved (only this model gets right)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(models, unique_solved, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, unique_solved):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel("Count", fontsize=12)
ax.set_title("Uniquely Solved Problems (only this model correct)", fontsize=14)
ax.set_ylim(0, 55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_unique_solved.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_unique_solved.png")

print(f"\nAll figures saved to {OUT_DIR}/")
