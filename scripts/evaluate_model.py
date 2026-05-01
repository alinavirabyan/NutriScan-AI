#!/usr/bin/env python3
"""
FULL EVALUATION - ALL test data
Base eng vs Fine-tuned eng_textocr
"""

import os
import sys
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FINAL_WORK = Path("/home/alina/DiplomaWork/Final_Work")
TEST_DIR = FINAL_WORK / "outputs" / "test_lstmf"
EVAL_DIR = FINAL_WORK / "outputs" / "evaluation_results"
CHARTS_DIR = EVAL_DIR / "evaluation_charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  FULL EVALUATION - ALL TEST DATA")
print("=" * 70)
print(f"\n📂 Loading test data...")

png_files = sorted(TEST_DIR.glob("*.png"))
valid = [f for f in png_files if f.with_suffix(".txt").exists()]
print(f"   Total test images: {len(valid):,}")
print(f"   This will take ~2 hours\n")
def cer(gt, pred):
    """Calculate Character Error Rate between ground truth and prediction.

    Uses Levenshtein distance normalised by ground truth length.
    Lower is better.

    Args:
        gt: Ground truth text.
        pred: Predicted text.

    Returns:
        Character Error Rate as a float between 0.0 and 1.0+.
    """
    if not gt: return 0.0
    a, b = list(gt.lower()), list(pred.lower())
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1] / len(a)

def wer(gt, pred):
    """Calculate Word Error Rate between ground truth and prediction.

    Uses Levenshtein distance on word-level tokens normalised by
    ground truth word count. Lower is better.

    Args:
        gt: Ground truth text.
        pred: Predicted text.

    Returns:
        Word Error Rate as a float between 0.0 and 1.0+.
    """
    gt_w, pred_w = gt.lower().split(), pred.lower().split()
    if not gt_w: return 0.0
    a, b = gt_w, pred_w
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1] / len(a)

def accuracy(gt, pred):
    """Calculate word-level accuracy.

    The fraction of ground truth words that exactly match the
    predicted words at the same position. Higher is better.

    Args:
        gt: Ground truth text.
        pred: Predicted text.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    gt_w, pred_w = gt.lower().split(), pred.lower().split()
    if not gt_w: return 1.0
    correct = sum(1 for i, w in enumerate(gt_w) if i < len(pred_w) and pred_w[i] == w)
    return correct / len(gt_w)

def f1_score(gt, pred):
    """Calculate F1 score for word-level prediction.

    Computes the harmonic mean of precision and recall on the set of
    unique words. Higher is better.

    Args:
        gt: Ground truth text.
        pred: Predicted text.

    Returns:
        F1 score as a float between 0.0 and 1.0.
    """
    gt_w = set(gt.lower().split())
    pred_w = set(pred.lower().split())
    if not pred_w or not gt_w: return 0.0
    p = len(gt_w & pred_w) / len(pred_w)
    r = len(gt_w & pred_w) / len(gt_w)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def exact_match(gt, pred):
    """Check if prediction exactly matches ground truth (case-insensitive).

    Args:
        gt: Ground truth text.
        pred: Predicted text.

    Returns:
        1.0 if the texts match exactly, 0.0 otherwise.
    """
    return 1.0 if gt.lower().strip() == pred.lower().strip() else 0.0


t0 = time.time()
results = []

for i, png_path in enumerate(tqdm(valid, desc="Evaluating")):
    gt_path = png_path.with_suffix(".txt")
    gt = gt_path.read_text(encoding="utf-8").strip()
    if not gt:
        continue
    
    try:
        base_pred = os.popen(f"tesseract {png_path} stdout --psm 7 -l eng 2>/dev/null").read().strip()
        ft_pred = os.popen(f"tesseract {png_path} stdout --psm 7 -l eng_textocr 2>/dev/null").read().strip()
        
        results.append({
            'image': png_path.name,
            'ground_truth': gt,
            'base_pred': base_pred,
            'ft_pred': ft_pred,
            'base_cer': cer(gt, base_pred),
            'ft_cer': cer(gt, ft_pred),
            'base_wer': wer(gt, base_pred),
            'ft_wer': wer(gt, ft_pred),
            'base_acc': accuracy(gt, base_pred),
            'ft_acc': accuracy(gt, ft_pred),
            'base_f1': f1_score(gt, base_pred),
            'ft_f1': f1_score(gt, ft_pred),
            'base_exact': exact_match(gt, base_pred),
            'ft_exact': exact_match(gt, ft_pred),
        })
    except:
        continue
    
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        print(f"\n   Progress: {i+1:,}/{len(valid):,} | "
              f"FT CER: {np.mean([r['ft_cer'] for r in results]):.1%} | "
              f"Elapsed: {elapsed/60:.0f}min")

elapsed = time.time() - t0
df = pd.DataFrame(results)
n = len(df)
m = {
    'base_cer': df['base_cer'].mean(), 'ft_cer': df['ft_cer'].mean(),
    'base_wer': df['base_wer'].mean(), 'ft_wer': df['ft_wer'].mean(),
    'base_acc': df['base_acc'].mean(), 'ft_acc': df['ft_acc'].mean(),
    'base_f1': df['base_f1'].mean(), 'ft_f1': df['ft_f1'].mean(),
    'base_exact': df['base_exact'].mean(), 'ft_exact': df['ft_exact'].mean(),
    'n': n,
}
m['cer_imp'] = (m['base_cer'] - m['ft_cer']) / m['base_cer'] * 100 if m['base_cer'] else 0
m['wer_imp'] = (m['base_wer'] - m['ft_wer']) / m['base_wer'] * 100 if m['base_wer'] else 0
m['acc_imp'] = (m['ft_acc'] - m['base_acc']) * 100
m['f1_imp'] = (m['ft_f1'] - m['base_f1']) * 100
m['exact_imp'] = (m['ft_exact'] - m['base_exact']) * 100

ft_better = (df['ft_cer'] < df['base_cer']).sum()
base_better = (df['base_cer'] < df['ft_cer']).sum()

print(f"\n{'='*70}")
print(f"  EVALUATION RESULTS ({n:,} samples, {elapsed/60:.0f} min)")
print(f"{'='*70}")
print(f"  {'Metric':<25} {'Base eng':>12} {'Fine-tuned':>12} {'Improvement':>12}")
print(f"  {'─'*65}")
print(f"  {'CER ↓':<25} {m['base_cer']:>11.1%} {m['ft_cer']:>11.1%} {m['cer_imp']:>+11.1f}%")
print(f"  {'WER ↓':<25} {m['base_wer']:>11.1%} {m['ft_wer']:>11.1%} {m['wer_imp']:>+11.1f}%")
print(f"  {'Accuracy ↑':<25} {m['base_acc']:>11.1%} {m['ft_acc']:>11.1%} {m['acc_imp']:>+11.1f}%")
print(f"  {'F1 Score ↑':<25} {m['base_f1']:>11.1%} {m['ft_f1']:>11.1%} {m['f1_imp']:>+11.1f}%")
print(f"  {'Exact Match ↑':<25} {m['base_exact']:>11.1%} {m['ft_exact']:>11.1%} {m['exact_imp']:>+11.1f}%")
print(f"{'='*70}")
print(f"\n  FT better: {ft_better:,} | Base better: {base_better:,}")

cer_ok = m['ft_cer'] <= 0.30
wer_ok = m['ft_wer'] <= 0.50
f1_ok = m['ft_f1'] >= 0.50
passed = sum([cer_ok, wer_ok, f1_ok])

print(f"\n  TARGET CHECK:")
print(f"    CER ≤ 30% : {'✅ PASS' if cer_ok else '❌ FAIL'} ({m['ft_cer']:.1%})")
print(f"    WER ≤ 50% : {'✅ PASS' if wer_ok else '❌ FAIL'} ({m['ft_wer']:.1%})")
print(f"    F1  ≥ 50% : {'✅ PASS' if f1_ok else '❌ FAIL'} ({m['ft_f1']:.1%})")
print(f"    {passed}/3 targets met")

print(f"\n📊 Generating charts...")
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(1)
ax1.bar(x - w/2, [m['base_cer']], w, label='Base eng', color='#EF5350')
ax1.bar(x + w/2, [m['ft_cer']], w, label='Fine-tuned', color='#42A5F5')
ax1.set_ylabel('CER (%)', fontweight='bold')
ax1.set_title('CER', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(['CER']); ax1.legend()
ax1.set_ylim(0, max(m['base_cer'], m['ft_cer']) * 1.3)

ax2.bar(x - w/2, [m['base_wer']], w, label='Base eng', color='#EF5350')
ax2.bar(x + w/2, [m['ft_wer']], w, label='Fine-tuned', color='#42A5F5')
ax2.set_ylabel('WER (%)', fontweight='bold')
ax2.set_title('WER', fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(['WER']); ax2.legend()
ax2.set_ylim(0, max(m['base_wer'], m['ft_wer']) * 1.3)

plt.suptitle('Սխալի Մակարդակի Համեմատություն', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "eval_cer_wer.png", dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
x2 = np.arange(2)
ax.bar(x2 - w/2, [m['base_acc'], m['base_f1']], w, label='Base eng', color='#EF5350')
ax.bar(x2 + w/2, [m['ft_acc'], m['ft_f1']], w, label='Fine-tuned', color='#42A5F5')
ax.set_ylabel('Ցուցանիշ (%)', fontweight='bold')
ax.set_title('Ճշտություն և F1 Գնահատական', fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(['Ճշտություն', 'F1']); ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "eval_accuracy_f1.png", dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
labels = ['CER', 'WER', 'Ճշտություն', 'F1', 'Exact Match']
vals = [m['cer_imp'], m['wer_imp'], m['acc_imp'], m['f1_imp'], m['exact_imp']]
colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]
ax.bar(labels, vals, color=colors, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylabel('Բարելավում (%)', fontweight='bold')
ax.set_title('Մոդելի Բարելավում', fontweight='bold')
for i, v in enumerate(vals):
    ax.text(i, v + 1 if v > 0 else v - 3, f'{v:+.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS_DIR / "eval_improvement.png", dpi=150, bbox_inches='tight')
plt.close()
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
data = [
    ['Ցուցանիշ', 'Base eng', 'Fine-tuned', 'Բարելավում'],
    ['CER ↓', f"{m['base_cer']:.1%}", f"{m['ft_cer']:.1%}", f"{m['cer_imp']:+.1f}%"],
    ['WER ↓', f"{m['base_wer']:.1%}", f"{m['ft_wer']:.1%}", f"{m['wer_imp']:+.1f}%"],
    ['Ճշտություն ↑', f"{m['base_acc']:.1%}", f"{m['ft_acc']:.1%}", f"{m['acc_imp']:+.1f}%"],
    ['F1 ↑', f"{m['base_f1']:.1%}", f"{m['ft_f1']:.1%}", f"{m['f1_imp']:+.1f}%"],
    ['Exact Match ↑', f"{m['base_exact']:.1%}", f"{m['ft_exact']:.1%}", f"{m['exact_imp']:+.1f}%"],
    ['Նմուշներ', str(n), str(n), '—'],
]
table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.25, 0.2, 0.2, 0.2])
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.2)
for j in range(4):
    table[(0, j)].set_facecolor('#37474F')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
ax.set_title('Գնահատման Ամփոփում', fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "eval_summary_table.png", dpi=150, bbox_inches='tight')
plt.close()

print("   ✅ All charts saved!")

df.to_csv(EVAL_DIR / "evaluation_results.csv", index=False)
with open(EVAL_DIR / "evaluation_summary.txt", 'w') as f:
    f.write(f"FULL EVALUATION ({n:,} samples, {elapsed/60:.0f} min)\n")
    f.write(f"{'='*50}\n")
    f.write(f"CER:  {m['base_cer']:.1%} → {m['ft_cer']:.1%} ({m['cer_imp']:+.1f}%)\n")
    f.write(f"WER:  {m['base_wer']:.1%} → {m['ft_wer']:.1%} ({m['wer_imp']:+.1f}%)\n")
    f.write(f"Acc:  {m['base_acc']:.1%} → {m['ft_acc']:.1%} ({m['acc_imp']:+.1f}%)\n")
    f.write(f"F1:   {m['base_f1']:.1%} → {m['ft_f1']:.1%} ({m['f1_imp']:+.1f}%)\n")
    f.write(f"Exact: {m['base_exact']:.1%} → {m['ft_exact']:.1%} ({m['exact_imp']:+.1f}%)\n")

print(f"\n{'='*70}")
print(f"  ✅ EVALUATION COMPLETE!")
print(f"{'='*70}")
print(f"  Results: {EVAL_DIR}")
print(f"  Charts:  {CHARTS_DIR}")
print(f"  {'='*70}")