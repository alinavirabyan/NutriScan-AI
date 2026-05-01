#!/usr/bin/env python3
"""
TESSERACT FINE-TUNING TRAINING - COMPLETE
Charts in ARMENIAN
"""

import os, re, subprocess, sys, time, shutil, logging
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
TRAIN_LIST = "/home/alina/DiplomaWork/Final_Work/outputs/train_lstmf/all.txt"
BASE_LSTM = "/home/alina/DiplomaWork/Diploma_work/models/tesseract/eng_best.lstm"
BASE_TRAINEDDATA = "/usr/share/tesseract-ocr/5/tessdata/eng_best.traineddata"
TESSDATA_DIR = "/usr/share/tesseract-ocr/5/tessdata"

FINAL_WORK = Path("/home/alina/DiplomaWork/Final_Work")
CHECKPOINT_DIR = FINAL_WORK / "outputs" / "checkpoints"
MODEL_DIR = FINAL_WORK / "models" / "tesseract"
CHARTS_DIR = FINAL_WORK / "outputs" / "evaluation_results" / "training_charts"

MODEL_NAME = "eng_textocr"
MAX_ITERATIONS = 1000000
LEARNING_RATE = 0.00002

for d in [CHECKPOINT_DIR, MODEL_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = CHECKPOINT_DIR / f"training_{RUN_STAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("  TESSERACT FINE-TUNING TRAINING")
print("=" * 70)
n_files = sum(1 for _ in open(TRAIN_LIST) if _.strip())
print(f"  Training files: {n_files:,}")
print(f"  Max iterations: {MAX_ITERATIONS:,}")
print(f"  Learning rate:  {LEARNING_RATE}")
print(f"  Started:        {datetime.now().strftime('%H:%M:%S')}")
print(f"  Est. time:      4-6 hours")
print("=" * 70)
print(f"\nTraining started...\n")

t_start = time.time()
iterations, bcer_vals, bwer_vals, delta_vals, rms_vals = [], [], [], [], []
best_bcer = 100.0

process = subprocess.Popen([
    "lstmtraining",
    f"--model_output={CHECKPOINT_DIR / MODEL_NAME}",
    f"--continue_from={BASE_LSTM}",
    f"--traineddata={BASE_TRAINEDDATA}",
    f"--train_listfile={TRAIN_LIST}",
    f"--max_iterations={MAX_ITERATIONS}",
    f"--learning_rate={LEARNING_RATE}",
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

for line in process.stdout:
    line = line.rstrip()
    logger.info(line)
    
    m_iter = re.search(r"At iteration\s+(\d+)", line)
    m_bcer = re.search(r"BCER train=([\d.]+)%", line)
    m_bwer = re.search(r"BWER train=([\d.]+)%", line)
    m_delta = re.search(r"delta=([\d.]+)%", line)
    m_rms = re.search(r"mean rms=([\d.]+)%", line)
    m_best = re.search(r"New best BCER = ([\d.]+)", line)
    
    if m_iter and m_bcer:
        iterations.append(int(m_iter.group(1)))
        bcer_vals.append(float(m_bcer.group(1)))
        if m_bwer: bwer_vals.append(float(m_bwer.group(1)))
        if m_delta: delta_vals.append(float(m_delta.group(1)))
        if m_rms: rms_vals.append(float(m_rms.group(1)))
    
    if m_best:
        best_bcer = float(m_best.group(1))
        elapsed = time.time() - t_start
        print(f"  BEST BCER={best_bcer:.3f}% | "
              f"Iter={iterations[-1]:,} | {elapsed/3600:.1f}h")

process.wait()
elapsed = time.time() - t_start

print(f"\nSaving metrics...")

metrics_file = CHECKPOINT_DIR / f"metrics_{RUN_STAMP}.txt"
with open(metrics_file, 'w') as f:
    f.write(f"# Tesseract Training Metrics\n")
    f.write(f"# Files: {n_files:,} | Iterations: {MAX_ITERATIONS:,} | LR: {LEARNING_RATE}\n")
    f.write(f"# Time: {elapsed/3600:.1f}h\n")
    f.write(f"# BCER: {bcer_vals[0]:.1f}% -> {bcer_vals[-1]:.1f}% | Best: {best_bcer:.3f}%\n")
    f.write(f"# iter,bcer,bwer,delta,rms\n")
    for i in range(len(iterations)):
        bw = bwer_vals[i] if i < len(bwer_vals) else 0
        dl = delta_vals[i] if i < len(delta_vals) else 0
        rm = rms_vals[i] if i < len(rms_vals) else 0
        f.write(f"{iterations[i]},{bcer_vals[i]:.3f},{bw:.3f},{dl:.3f},{rm:.3f}\n")

print(f"   Done: {metrics_file}")

checkpoints = sorted(CHECKPOINT_DIR.glob("*.checkpoint"))
if not checkpoints:
    print("ERROR: No checkpoints!")
    sys.exit(1)

def bcer_of(p):
    m = re.search(r"_([\d.]+)_", p.name)
    return float(m.group(1)) if m else 999.0

best = min(checkpoints, key=bcer_of)
final_bcer = bcer_of(best)

output_model = MODEL_DIR / f"{MODEL_NAME}.traineddata"
subprocess.run(["lstmtraining", "--stop_training",
    f"--continue_from={best}", f"--traineddata={BASE_TRAINEDDATA}",
    f"--model_output={output_model}"], check=True)

dest = Path(TESSDATA_DIR) / f"{MODEL_NAME}.traineddata"
try:
    shutil.copy2(output_model, dest)
    print(f"Model installed to {dest}")
except PermissionError:
    print(f"Run: sudo cp {output_model} {dest}")




print(f"\nGenerating training charts...")


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(iterations, bcer_vals, linewidth=2, color='#FF7043')
ax.fill_between(iterations, bcer_vals, alpha=0.1, color='#FF7043')
ax.scatter([iterations[0]], [bcer_vals[0]], color='red', s=100, zorder=5, 
           label=f'Սկիզբ: {bcer_vals[0]:.1f}%')
ax.scatter([iterations[-1]], [bcer_vals[-1]], color='green', s=100, zorder=5, 
           label=f'Վերջ: {bcer_vals[-1]:.1f}%')
ax.set_xlabel('Իտերացիաներ', fontweight='bold', fontsize=12)
ax.set_ylabel('BCER (%)', fontweight='bold', fontsize=12)
ax.set_title(f'Ուսուցման Կոր: BCER ({bcer_vals[0]:.1f}% → {bcer_vals[-1]:.1f}%)', 
             fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "chart1_bcer_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("   chart1_bcer_curve.png")

if bwer_vals:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, bwer_vals, linewidth=2, color='#AB47BC')
    ax.fill_between(iterations, bwer_vals, alpha=0.1, color='#AB47BC')
    ax.scatter([iterations[0]], [bwer_vals[0]], color='red', s=100, zorder=5,
               label=f'Սկիզբ: {bwer_vals[0]:.1f}%')
    ax.scatter([iterations[-1]], [bwer_vals[-1]], color='green', s=100, zorder=5,
               label=f'Վերջ: {bwer_vals[-1]:.1f}%')
    ax.set_xlabel('Իտերացիաներ', fontweight='bold', fontsize=12)
    ax.set_ylabel('BWER (%)', fontweight='bold', fontsize=12)
    ax.set_title('Ուսուցման Կոր: BWER (Բառերի Սխալի Մակարդակ)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "chart2_bwer_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   chart2_bwer_curve.png")

if bwer_vals:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, bcer_vals, '-', linewidth=2, label='BCER (Նիշերի սխալ)', color='#FF7043')
    ax.plot(iterations, bwer_vals, '--', linewidth=2, label='BWER (Բառերի սխալ)', color='#AB47BC')
    ax.set_xlabel('Իտերացիաներ', fontweight='bold', fontsize=12)
    ax.set_ylabel('Սխալի Մակարդակ (%)', fontweight='bold', fontsize=12)
    ax.set_title('BCER և BWER Ուսուցման Ընթացքում', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "chart3_bcer_bwer_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   chart3_bcer_bwer_combined.png")

if delta_vals:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, delta_vals, linewidth=2, color='#42A5F5')
    ax.fill_between(iterations, delta_vals, alpha=0.1, color='#42A5F5')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Կայունացած (<1%)')
    ax.set_xlabel('Իտերացիաներ', fontweight='bold', fontsize=12)
    ax.set_ylabel('Դելտա (%)', fontweight='bold', fontsize=12)
    ax.set_title('Մոդելի Կայունացում: Դելտա', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "chart4_delta_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   chart4_delta_convergence.png")

fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(iterations, bcer_vals, '-', linewidth=2, label='BCER (Նիշերի սխալ)', color='#FF7043')
if bwer_vals: 
    ax1.plot(iterations, bwer_vals, '--', linewidth=2, label='BWER (Բառերի սխալ)', color='#AB47BC')
ax1.set_xlabel('Իտերացիաներ', fontweight='bold', fontsize=12)
ax1.set_ylabel('Սխալի Մակարդակ (%)', fontweight='bold', fontsize=12, color='#FF7043')
ax1.tick_params(axis='y', labelcolor='#FF7043')

if delta_vals:
    ax2 = ax1.twinx()
    ax2.plot(iterations, delta_vals, '-', linewidth=1, alpha=0.5, 
             label='Դելտա (Քաշերի փոփոխություն)', color='#42A5F5')
    ax2.set_ylabel('Դելտա (%)', fontweight='bold', fontsize=12, color='#42A5F5')
    ax2.tick_params(axis='y', labelcolor='#42A5F5')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels() if delta_vals else ([], [])
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
ax1.set_title(f'Ուսուցման Ամփոփում: BCER {bcer_vals[0]:.1f}% → {bcer_vals[-1]:.1f}%', 
              fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "chart5_all_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("   chart5_all_metrics.png")

print(f"\n{'='*70}")
print(f"  TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"""
  Time:     {elapsed/3600:.1f}h
  BCER:     {bcer_vals[0]:.1f}% -> {bcer_vals[-1]:.1f}% (best: {final_bcer:.3f}%)
  BWER:     {bwer_vals[0]:.1f}% -> {bwer_vals[-1]:.1f}% (if available)
  Delta:    {delta_vals[0]:.1f}% -> {delta_vals[-1]:.1f}% (if available)
  
  OUTPUT:
  Model:    {output_model}
  Charts:   {CHARTS_DIR}
  Log:      {log_file}
  Metrics:  {metrics_file}
  
  Charts generated:
  chart1_bcer_curve.png
  chart2_bwer_curve.png
  chart3_bcer_bwer_combined.png
  chart4_delta_convergence.png
  chart5_all_metrics.png
""")
print("=" * 70)