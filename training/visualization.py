# training/visualization.py
"""
Visualization functions for OCR training pipeline
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def chart_dataset_overview(train_count, val_count, test_count,
                           total_annot, clean_annot, used_annot, 
                           charts_dir):
    """Create dataset overview chart"""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Dataset Overview", fontsize=15, fontweight="bold", y=1.02)

    # Pie chart - Image split
    ax = axes[0]
    sizes = [train_count, val_count, test_count]
    labels = [f"Train\n{train_count:,}", f"Val\n{val_count:,}", f"Test\n{test_count:,}"]
    colors = ["#42A5F5", "#FFA726", "#66BB6A"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title(f"Image Split\n({sum(sizes):,} total images)",
                 fontsize=12, fontweight="bold")

    # Bar chart - Annotation pipeline
    ax2 = axes[1]
    stages = ["Total\nannotations", "After\ncleaning", "Used for\ntraining"]
    values = [total_annot, clean_annot, used_annot]
    colors2 = ["#EF5350", "#FFA726", "#42A5F5"]
    bars = ax2.bar(stages, values, color=colors2, edgecolor="white", width=0.4)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + total_annot * 0.01,
                 f"{val:,}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    
    ax2.set_title("Annotation Pipeline", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of annotations", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_facecolor("#f9f9f9")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    fig.tight_layout()
    path = charts_dir / "1_dataset_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {path}")


def chart_training_curve(iterations, bcer_values, skip_ratios, charts_dir):
    """Create training curve chart"""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    if not iterations:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # BCER chart
    ax1 = axes[0]
    ax1.plot(iterations, bcer_values, 'b-', linewidth=2, label='BCER')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('BCER (%)')
    ax1.set_title('Training Curve - BCER (lower is better)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark best point
    min_bcer = min(bcer_values)
    min_idx = bcer_values.index(min_bcer)
    ax1.plot(iterations[min_idx], min_bcer, 'ro', markersize=8)
    ax1.annotate(f'Best: {min_bcer:.2f}%', 
                 xy=(iterations[min_idx], min_bcer),
                 xytext=(10, 10), textcoords='offset points')
    
    # Skip ratio chart
    if skip_ratios:
        ax2 = axes[1]
        ax2.plot(iterations[:len(skip_ratios)], skip_ratios, 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Skip Ratio (%)')
        ax2.set_title('Skipped Samples (should be <10%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=10, color='orange', linestyle='--', label='Warning threshold')
        ax2.legend()
    
    plt.tight_layout()
    path = charts_dir / "2_training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Chart saved: {path}")


def chart_cer_wer(base_cer, ft_cer, base_wer, ft_wer, count, charts_dir):
    """Create CER/WER comparison chart"""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Model Evaluation — Base vs Fine-tuned\n"
                 f"({count} test samples)",
                 fontsize=14, fontweight="bold")

    # Error rates chart
    ax = axes[0]
    metrics = ["CER", "WER"]
    base_vals = [base_cer, base_wer]
    ft_vals = [ft_cer, ft_wer]
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_vals, width, label="Base eng",
                   color=["#EF5350", "#EF9A9A"], edgecolor="white")
    bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tuned",
                   color=["#42A5F5", "#90CAF9"], edgecolor="white")

    for bar, val in zip(list(bars1) + list(bars2), base_vals + ft_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Error Rates (lower = better)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(base_vals) * 1.35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#f9f9f9")

    # Improvement chart
    ax2 = axes[1]
    cer_imp = (base_cer - ft_cer) / base_cer * 100 if base_cer else 0
    wer_imp = (base_wer - ft_wer) / base_wer * 100 if base_wer else 0
    imps = [cer_imp, wer_imp]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in imps]
    bars = ax2.bar(metrics, imps, color=colors, edgecolor="white", width=0.4)

    for bar, val in zip(bars, imps):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if val >= 0 else -1.0),
                 f"{val:+.1f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    ax2.set_title("Improvement % (higher = better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Improvement %", fontsize=11)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_facecolor("#f9f9f9")

    fig.tight_layout()
    path = charts_dir / "3_cer_wer_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Chart saved: {path}")


def chart_error_distribution(base_cers, ft_cers, charts_dir):
    """Create error distribution histogram"""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, 2, 40)

    ax.hist(base_cers, bins=bins, alpha=0.6, color="#EF5350",
            label=f"Base eng  (mean={np.mean(base_cers):.3f})",
            edgecolor="white")
    ax.hist(ft_cers, bins=bins, alpha=0.6, color="#42A5F5",
            label=f"Fine-tuned (mean={np.mean(ft_cers):.3f})",
            edgecolor="white")

    ax.axvline(np.mean(base_cers), color="#C62828", linestyle="--", linewidth=2)
    ax.axvline(np.mean(ft_cers), color="#1565C0", linestyle="--", linewidth=2)

    ax.set_title("CER Distribution per Sample\n"
                 "(shifted left = better — fewer errors)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CER per sample (lower = better)", fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f9f9f9")

    fig.tight_layout()
    path = charts_dir / "4_error_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Chart saved: {path}")


if __name__ == "__main__":
    print("Visualization module ready")
    print("Functions available:")
    print("  - chart_dataset_overview()")
    print("  - chart_training_curve()")
    print("  - chart_cer_wer()")
    print("  - chart_error_distribution()")