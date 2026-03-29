# training/evaluation.py
"""
Evaluation module for OCR training pipeline
"""

import os
import ast
import logging
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from training.config import *
from training.utils import compute_cer, compute_wer, banner
from training.visualization import chart_cer_wer, chart_error_distribution

logger = logging.getLogger(__name__)


def evaluate_on_crops(crops_dir: str, model_path: str, num_samples: int = 500) -> dict:
    """
    Evaluate fine-tuned model on word crops.
    Compares base eng vs fine-tuned model.

    Args:
        crops_dir:   path to .tif crop images
        model_path:  path to fine-tuned .traineddata
        num_samples: how many samples to evaluate

    Returns:
        dict with CER, WER metrics
    """
    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not installed. Run: pip install pytesseract")
        return {}

    # Install model to system tessdata if needed
    dest = Path(SYSTEM_TESSDATA) / f"{MODEL_NAME}.traineddata"
    if not dest.exists():
        try:
            import shutil
            shutil.copy2(model_path, dest)
            logger.info(f"Installed model to {dest}")
        except PermissionError:
            logger.error(f"Permission denied. Run: sudo cp {model_path} {dest}")
            return {}

    crops_dir  = Path(crops_dir)
    tif_files  = sorted(crops_dir.glob("*.tif"))[:num_samples]
    logger.info(f"Evaluating on {len(tif_files)} training samples ...")

    base_cer_sum = ft_cer_sum = 0.0
    base_wer_sum = ft_wer_sum = 0.0
    base_cers, ft_cers = [], []
    count = 0
    cfg   = "--psm 7"

    for tif_path in tqdm(tif_files):
        gt_path = tif_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            continue
        gt = gt_path.read_text(encoding="utf-8").strip()
        if not gt:
            continue
        try:
            base_pred = pytesseract.image_to_string(
                Image.open(tif_path), lang="eng",      config=cfg).strip()
            ft_pred   = pytesseract.image_to_string(
                Image.open(tif_path), lang=MODEL_NAME, config=cfg).strip()

            bc = compute_cer(gt, base_pred)
            fc = compute_cer(gt, ft_pred)
            base_cer_sum += bc
            ft_cer_sum   += fc
            base_wer_sum += compute_wer(gt, base_pred)
            ft_wer_sum   += compute_wer(gt, ft_pred)
            base_cers.append(bc)
            ft_cers.append(fc)
            count += 1
        except Exception as e:
            if count < 5:
                logger.warning(f"Error: {e}")
            continue

    if count == 0:
        logger.warning("No samples evaluated.")
        return {}

    base_cer = base_cer_sum / count
    ft_cer   = ft_cer_sum   / count
    base_wer = base_wer_sum / count
    ft_wer   = ft_wer_sum   / count
    cer_imp  = (base_cer - ft_cer) / base_cer * 100 if base_cer else 0
    wer_imp  = (base_wer - ft_wer) / base_wer * 100 if base_wer else 0

    _print_results(base_cer, ft_cer, base_wer, ft_wer, cer_imp, wer_imp, count)

    chart_cer_wer(base_cer, ft_cer, base_wer, ft_wer, count, CHARTS_DIR)
    chart_error_distribution(base_cers, ft_cers, CHARTS_DIR)

    return {
        "base_cer": base_cer, "ft_cer": ft_cer,
        "base_wer": base_wer, "ft_wer": ft_wer,
        "cer_improvement": cer_imp,
        "wer_improvement": wer_imp,
        "samples": count,
    }


def evaluate_on_test_split(model_path: str, num_samples: int = 300) -> dict:
    """
    Evaluate on the TEST split — images never seen during training.
    This gives honest accuracy metrics.

    Args:
        model_path:  path to fine-tuned .traineddata
        num_samples: how many test annotations to evaluate

    Returns:
        dict with CER, WER metrics
    """
    banner("EVALUATING ON TEST SPLIT (unseen data)")

    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not installed.")
        return {}

    # Install model
    dest = Path(SYSTEM_TESSDATA) / f"{MODEL_NAME}.traineddata"
    if not dest.exists():
        try:
            import shutil
            shutil.copy2(model_path, dest)
        except PermissionError:
            logger.error(f"Run: sudo cp {model_path} {dest}")
            return {}

    # Load test split
    test_img   = pd.read_csv(SPLITS_DIR / "test_img.csv",   index_col=0)
    test_annot = pd.read_csv(SPLITS_DIR / "test_annot.csv", index_col=0)

    logger.info(f"Test images      : {len(test_img):,}")
    logger.info(f"Test annotations : {len(test_annot):,}")

    # Clean
    test_annot = test_annot[test_annot["utf8_string"].notna()]
    test_annot = test_annot[test_annot["utf8_string"].astype(str).str.len() >= MIN_TEXT_LEN]
    test_annot = test_annot[
        test_annot["utf8_string"].astype(str).str.contains(r'[a-zA-Z]', regex=True)
    ]
    test_annot = test_annot[
        test_annot["utf8_string"].astype(str).apply(
            lambda x: x.encode('ascii', errors='ignore').decode() == x
        )
    ]

    if len(test_annot) > num_samples:
        test_annot = test_annot.sample(n=num_samples, random_state=RANDOM_STATE)

    logger.info(f"Using {len(test_annot):,} test samples")

    images_folder = DATASET_PATH / "train_val_images" / "train_images"
    id_to_file    = dict(zip(test_img["id"], test_img["file_name"]))
    img_cache     = {}
    cfg           = "--psm 7"

    base_cers, ft_cers = [], []
    base_wers, ft_wers = [], []
    count = 0

    for _, row in tqdm(test_annot.iterrows(), total=len(test_annot)):
        try:
            image_id = row["image_id"]
            gt_text  = str(row["utf8_string"]).strip()
            bbox     = ast.literal_eval(row["bbox"]) \
                       if isinstance(row["bbox"], str) else list(row["bbox"])

            fname = id_to_file.get(image_id)
            if not fname:
                continue
            fname    = Path(fname).name
            img_path = images_folder / fname
            if not img_path.exists():
                continue

            if image_id not in img_cache:
                if len(img_cache) > 50:
                    img_cache.clear()
                img_cache[image_id] = Image.open(img_path).convert("RGB")
            img = img_cache[image_id]

            x, y, w, h = [int(float(v)) for v in bbox]
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(img.width,  x + w + PADDING)
            y2 = min(img.height, y + h + PADDING)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            from PIL import ImageOps
            crop = img.crop((x1, y1, x2, y2)).convert("L")
            crop = ImageOps.autocontrast(crop)
            if crop.height < 32:
                scale = 32 / crop.height
                crop  = crop.resize((int(crop.width * scale), 32), Image.LANCZOS)

            base_pred = pytesseract.image_to_string(
                crop, lang="eng",      config=cfg).strip()
            ft_pred   = pytesseract.image_to_string(
                crop, lang=MODEL_NAME, config=cfg).strip()

            base_cers.append(compute_cer(gt_text, base_pred))
            ft_cers.append(compute_cer(gt_text, ft_pred))
            base_wers.append(compute_wer(gt_text, base_pred))
            ft_wers.append(compute_wer(gt_text, ft_pred))
            count += 1

        except Exception:
            continue

    if count == 0:
        logger.warning("No test samples evaluated.")
        return {}

    base_cer = np.mean(base_cers)
    ft_cer   = np.mean(ft_cers)
    base_wer = np.mean(base_wers)
    ft_wer   = np.mean(ft_wers)
    cer_imp  = (base_cer - ft_cer) / base_cer * 100 if base_cer else 0
    wer_imp  = (base_wer - ft_wer) / base_wer * 100 if base_wer else 0

    _print_results(base_cer, ft_cer, base_wer, ft_wer, cer_imp, wer_imp,
                   count, title="TEST SET (unseen data)")

    # Save test charts separately
    chart_cer_wer(base_cer, ft_cer, base_wer, ft_wer, count,
                  CHARTS_DIR, prefix="test_")
    chart_error_distribution(base_cers, ft_cers,
                              CHARTS_DIR, prefix="test_")

    return {
        "base_cer": base_cer, "ft_cer": ft_cer,
        "base_wer": base_wer, "ft_wer": ft_wer,
        "cer_improvement": cer_imp,
        "wer_improvement": wer_imp,
        "samples": count,
        "evaluation_set": "test",
    }


def _print_results(base_cer, ft_cer, base_wer, ft_wer,
                   cer_imp, wer_imp, count, title="RESULTS"):
    print(f"\n  {'='*70}")
    print(f"  {title}")
    print(f"  {'Metric':<25} {'Base eng':>10} {'Fine-tuned':>12} {'Improvement':>13}")
    print(f"  {'─'*62}")
    print(f"  {'CER (lower=better)':<25} {base_cer:>9.3f}  {ft_cer:>11.3f}  {cer_imp:>+11.1f}%")
    print(f"  {'WER (lower=better)':<25} {base_wer:>9.3f}  {ft_wer:>11.3f}  {wer_imp:>+11.1f}%")
    print(f"  {'─'*62}")
    print(f"  Samples evaluated : {count}")
    print(f"  {'='*70}\n")