# training/train_pipeline.py
"""
Main OCR Training Pipeline
"""

import os
import ast
import re
import shutil
import subprocess
import logging
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import kagglehub

# Import from our modules
from training.config import *
from training.utils import *
from training.visualization import *

logger = logging.getLogger(__name__)


def step1_download() -> str:
    """Download TextOCR dataset"""
    banner("STEP 1: Downloading Dataset")
    
    # Check if dataset already exists
    if DATASET_PATH.exists() and (DATASET_PATH / "img.csv").exists():
        logger.info(f"Dataset already exists at {DATASET_PATH}")
        return str(DATASET_PATH)
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(
            "robikscube/textocr-text-extraction-from-images-dataset"
        )
        logger.info(f"Dataset downloaded to cache: {path}")
        
        # Copy to project directory
        logger.info(f"Copying to project: {DATASET_PATH}")
        for item in Path(path).glob("*"):
            if item.is_file():
                shutil.copy2(item, DATASET_PATH / item.name)
            elif item.is_dir():
                shutil.copytree(item, DATASET_PATH / item.name, dirs_exist_ok=True)
        
        logger.info(f"✅ Dataset saved to: {DATASET_PATH}")
        return str(DATASET_PATH)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Please download manually from: https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset")
        raise


def step2_split(dataset_path: str) -> str:
    """Split dataset into train/val/test"""
    banner("STEP 2: Splitting Dataset")

    dataset_path = Path(dataset_path)
    
    img_df = pd.read_csv(dataset_path / "img.csv", index_col=0)
    annot_df = pd.read_csv(dataset_path / "annot.csv", index_col=0)

    logger.info(f"Total images      : {len(img_df):,}")
    logger.info(f"Total annotations : {len(annot_df):,}")

    # Split images
    image_ids = img_df["id"].to_numpy()
    train_ids, temp_ids = train_test_split(
        image_ids, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_STATE
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
    )

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        img_df[img_df["id"].isin(ids)].to_csv(
            SPLITS_DIR / f"{name}_img.csv"
        )
        annot_df[annot_df["image_id"].isin(ids)].to_csv(
            SPLITS_DIR / f"{name}_annot.csv"
        )
        logger.info(f"  {name:5s}: {len(ids):,} images")

    # Calculate statistics for chart
    train_annot = annot_df[annot_df["image_id"].isin(train_ids)]
    clean_count = int(train_annot["utf8_string"].notna().sum())
    used_count = min(MAX_SAMPLES, clean_count) if MAX_SAMPLES else clean_count
    
    chart_dataset_overview(
        len(train_ids), len(val_ids), len(test_ids),
        len(annot_df), clean_count, used_count,
        charts_dir=CHARTS_DIR
    )

    logger.info(f"Saved to: {SPLITS_DIR}")
    return str(SPLITS_DIR)


def step3_prepare_crops(dataset_path: str, splits_dir: str) -> str:
    """Prepare cropped word images"""
    banner("STEP 3: Cleaning + Cropping Word Regions")

    dataset_path = Path(dataset_path)
    splits_dir = Path(splits_dir)
    
    # Find images folder
    images_folder = dataset_path / "train_val_images" / "train_images"
    
    if not images_folder.exists():
        images_folder = dataset_path / "train_images"
        
    if not images_folder.exists():
        for root, dirs, files in os.walk(dataset_path):
            if "train_images" in dirs:
                images_folder = Path(root) / "train_images"
                break
    
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found in {dataset_path}")

    # Load data
    img_df = pd.read_csv(splits_dir / "train_img.csv", index_col=0)
    annot_df = pd.read_csv(splits_dir / "train_annot.csv", index_col=0)

    logger.info(f"Train images : {len(img_df):,}")
    logger.info(f"Train annot  : {len(annot_df):,}")

    # Clean annotations
    before = len(annot_df)
    annot_df = annot_df[annot_df["utf8_string"].notna()]
    annot_df = annot_df[
        annot_df["utf8_string"].astype(str).apply(is_clean_label)
    ]
    logger.info(f"After cleaning : {len(annot_df):,} (removed {before - len(annot_df):,})")

    # Limit samples if specified
    if MAX_SAMPLES and len(annot_df) > MAX_SAMPLES:
        annot_df = annot_df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
    logger.info(f"Using          : {len(annot_df):,} samples")

    # Process each annotation
    id_to_file = dict(zip(img_df["id"], img_df["file_name"]))
    img_cache = {}
    saved = 0
    skipped = {"no_match": 0, "no_file": 0, "bad_crop": 0, "error": 0}

    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    logger.info(f"Processing with: {'GPU' if use_gpu else 'CPU'}")

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(annot_df.iterrows(), total=len(annot_df)):
        try:
            image_id = row["image_id"]
            text = str(row["utf8_string"]).strip()
            bbox = ast.literal_eval(row["bbox"]) \
                   if isinstance(row["bbox"], str) else list(row["bbox"])

            fname = id_to_file.get(image_id)
            if not fname:
                skipped["no_match"] += 1
                continue

            fname = Path(fname).name
            img_path = images_folder / fname
            if not img_path.exists():
                skipped["no_file"] += 1
                continue

            if image_id not in img_cache:
                if len(img_cache) > 100:
                    img_cache.clear()
                img_cache[image_id] = Image.open(img_path).convert("RGB")
            img = img_cache[image_id]

            x, y, w, h = [int(float(v)) for v in bbox]
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(img.width, x + w + PADDING)
            y2 = min(img.height, y + h + PADDING)

            if x2 - x1 < 8 or y2 - y1 < 8:
                skipped["bad_crop"] += 1
                continue

            if use_gpu and img.width * img.height > 500000:
                crop = preprocess_crop_gpu(img.crop((x1, y1, x2, y2)))
            else:
                crop = preprocess_crop(img.crop((x1, y1, x2, y2)))
                
            stem = f"word_{saved:06d}"
            crop.save(TRAINING_DATA_DIR / f"{stem}.tif", format="TIFF")
            with open(TRAINING_DATA_DIR / f"{stem}.gt.txt", "w", encoding="utf-8") as f:
                f.write(text)
            saved += 1

        except Exception as e:
            skipped["error"] += 1
            if skipped["error"] <= 3:
                logger.warning(f"Crop error: {e}")

    logger.info(f"Saved   : {saved:,} word crops")
    logger.info(f"Skipped : {sum(skipped.values()):,} → {skipped}")
    return str(TRAINING_DATA_DIR)


def step4_generate_lstmf(crops_dir: str) -> str:
    """Generate .lstmf files for Tesseract training"""
    banner("STEP 4: Generating .lstmf Files")

    crops_dir = Path(crops_dir)
    LSTMF_DIR.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(crops_dir.glob("*.tif"))
    lstmf_paths = []
    success, failed = 0, 0

    if not tif_files:
        raise RuntimeError(f"No TIFF files found in {crops_dir}")

    logger.info(f"Converting {len(tif_files):,} crops to .lstmf ...")

    for tif_path in tqdm(tif_files):
        gt_path = tif_path.with_suffix(".gt.txt")
        box_path = tif_path.with_suffix(".box")

        if not gt_path.exists():
            failed += 1
            continue

        try:
            text = gt_path.read_text(encoding="utf-8").strip()
            img = Image.open(tif_path)
            w, h = img.size
            char_w = max(1, w // max(len(text), 1))
            with open(box_path, "w", encoding="utf-8") as bf:
                for i, ch in enumerate(text):
                    x1 = i * char_w
                    x2 = min(x1 + char_w, w)
                    bf.write(f"{ch} {x1} 0 {x2} {h} 0\n")
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning(f"Box error: {e}")
            continue

        lstmf_path = LSTMF_DIR / f"{tif_path.stem}.lstmf"
        result = subprocess.run([
            "tesseract", str(tif_path),
            str(LSTMF_DIR / tif_path.stem),
            "--psm", "7",
            "-l", BASE_LANG,
            "lstm.train",
        ], capture_output=True, text=True)

        if result.returncode == 0 and lstmf_path.exists():
            lstmf_paths.append(str(lstmf_path))
            success += 1
        else:
            failed += 1
            if failed <= 3:
                logger.warning(f"Tesseract failed for {tif_path.name}: {result.stderr}")

    list_file = LSTMF_DIR / "all.txt"
    with open(list_file, "w") as f:
        f.write("\n".join(lstmf_paths))

    logger.info(f"Generated : {success:,} | Failed : {failed:,}")

    if success == 0:
        raise RuntimeError("0 .lstmf files generated!")
    return str(LSTMF_DIR)


def step5_train(lstmf_dir: str) -> str:
    """Fine-tune Tesseract model"""
    banner("STEP 5: Fine-tuning Tesseract LSTM")

    lstmf_dir = Path(lstmf_dir)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    base_lstm = MODEL_DIR / f"{BASE_LANG}.lstm"
    list_file = lstmf_dir / "all.txt"

    if not list_file.exists():
        raise RuntimeError(f"all.txt not found at {list_file}")

    # Count training files
    with open(list_file) as f:
        n_files = sum(1 for line in f if line.strip())
    if n_files == 0:
        raise RuntimeError("all.txt is empty!")
    logger.info(f"Training on {n_files:,} .lstmf files")

    # Extract base LSTM if needed
    if not base_lstm.exists():
        logger.info("Extracting base LSTM ...")
        try:
            subprocess.run([
                "combine_tessdata", "-e",
                os.path.join(TESSDATA_DIR, f"{BASE_LANG}.traineddata"),
                str(base_lstm),
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract base LSTM: {e.stderr}")
            raise

    logger.info(f"Training for {MAX_ITERATIONS} iterations (lr={LEARNING_RATE})")
    logger.info("BCER printed every 100 iterations — lower = better\n")

    # Run training
    process = subprocess.Popen([
        "lstmtraining",
        f"--traineddata={os.path.join(TESSDATA_DIR, BASE_LANG + '.traineddata')}",
        f"--model_output={CHECKPOINT_DIR / MODEL_NAME}",
        f"--continue_from={base_lstm}",
        f"--learning_rate={LEARNING_RATE}",
        f"--train_listfile={list_file}",
        f"--max_iterations={MAX_ITERATIONS}",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Collect metrics for chart
    iterations = []
    bcer_values = []
    skip_ratios = []

    for line in process.stdout:
        logger.info(line.strip())
        
        # Parse metrics
        if "BCER train=" in line:
            try:
                match = re.search(r'BCER train=([0-9.]+)%', line)
                if match:
                    bcer = float(match.group(1))
                    
                    iter_match = re.search(r'iteration (\d+)', line)
                    if iter_match:
                        iterations.append(int(iter_match.group(1)))
                        bcer_values.append(bcer)
                        
                        skip_match = re.search(r'skip ratio=([0-9.]+)%', line)
                        if skip_match:
                            skip_ratios.append(float(skip_match.group(1)))
            except:
                pass

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("Training failed")

    # Create training curve chart
    chart_training_curve(iterations, bcer_values, skip_ratios, CHARTS_DIR)

    # Find best checkpoint
    logger.info("\nFinding best checkpoint ...")
    checkpoints = list(CHECKPOINT_DIR.glob(f"{MODEL_NAME}_*.checkpoint"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("*.checkpoint"))
    
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found! Training failed.")
    
    def extract_bcer(p: Path) -> float:
        m = re.search(rf"{MODEL_NAME}_([0-9.]+)_", p.name)
        return float(m.group(1)) if m else 999.0

    best = min(checkpoints, key=extract_bcer)
    best_bcer = extract_bcer(best)
    logger.info(f"Best checkpoint : {best.name}  (BCER={best_bcer:.3f}%)")

    # Convert to final model
    output_model = MODEL_DIR / f"{MODEL_NAME}.traineddata"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    subprocess.run([
        "lstmtraining", "--stop_training",
        f"--continue_from={best}",
        f"--traineddata={os.path.join(TESSDATA_DIR, BASE_LANG + '.traineddata')}",
        f"--model_output={output_model}",
    ], check=True)

    logger.info(f"✅ Model saved : {output_model}")
    logger.info(f"   Best BCER  : {best_bcer:.3f}%")
    logger.info(f"   Model size : {output_model.stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(output_model)


def step6_evaluate(crops_dir: str, model_path: str) -> None:
    """Evaluate trained model"""
    banner("STEP 6: Evaluating Model — CER & WER + Charts")

    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not installed. Run: pip install pytesseract")
        return

    # Copy model to system if needed
    dest = Path(SYSTEM_TESSDATA) / f"{MODEL_NAME}.traineddata"
    if not dest.exists():
        try:
            shutil.copy2(model_path, dest)
            logger.info(f"Installed model to {dest}")
        except PermissionError:
            logger.error(f"Permission denied. Run: sudo cp {model_path} {dest}")

    # Find test files
    crops_dir = Path(crops_dir)
    tif_files = sorted(crops_dir.glob("*.tif"))[:500]  # Limit to 500 for speed
    logger.info(f"Evaluating on {len(tif_files)} samples ...")

    base_cer_sum = ft_cer_sum = 0.0
    base_wer_sum = ft_wer_sum = 0.0
    base_cers = []
    ft_cers = []
    count = 0

    cfg = "--psm 7"
    
    for tif_path in tqdm(tif_files):
        gt_path = tif_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            continue
        
        gt = gt_path.read_text(encoding="utf-8").strip()
        if not gt:
            continue
            
        try:
            # Base model
            base_pred = pytesseract.image_to_string(
                Image.open(tif_path), lang="eng", config=cfg).strip()
            
            # Fine-tuned model
            ft_pred = pytesseract.image_to_string(
                Image.open(tif_path), lang=MODEL_NAME, config=cfg).strip()

            bc = compute_cer(gt, base_pred)
            fc = compute_cer(gt, ft_pred)
            
            base_cer_sum += bc
            ft_cer_sum += fc
            base_wer_sum += compute_wer(gt, base_pred)
            ft_wer_sum += compute_wer(gt, ft_pred)
            base_cers.append(bc)
            ft_cers.append(fc)
            count += 1
            
        except Exception as e:
            if count < 5:
                logger.warning(f"Evaluation error for {tif_path.name}: {e}")
            continue

    if count == 0:
        logger.warning("No samples evaluated.")
        return

    # Calculate averages
    base_cer = base_cer_sum / count
    ft_cer = ft_cer_sum / count
    base_wer = base_wer_sum / count
    ft_wer = ft_wer_sum / count
    
    cer_improvement = (base_cer - ft_cer) / base_cer * 100 if base_cer else 0
    wer_improvement = (base_wer - ft_wer) / base_wer * 100 if base_wer else 0

    # Print results
    print(f"\n  {'='*70}")
    print(f"  {'Metric':<25} {'Base eng':>10} {'Fine-tuned':>12} {'Improvement':>13}")
    print(f"  {'─'*62}")
    print(f"  {'CER (lower=better)':<25} {base_cer:>9.3f}  {ft_cer:>11.3f}  {cer_improvement:>+11.1f}%")
    print(f"  {'WER (lower=better)':<25} {base_wer:>9.3f}  {ft_wer:>11.3f}  {wer_improvement:>+11.1f}%")
    print(f"  {'─'*62}")
    print(f"  Samples : {count}")
    print(f"  {'='*70}\n")

    # Create charts
    try:
        chart_cer_wer(base_cer, ft_cer, base_wer, ft_wer, count, CHARTS_DIR)
        chart_error_distribution(base_cers, ft_cers, CHARTS_DIR)
    except Exception as e:
        logger.warning(f"Could not generate charts: {e}")


def run_full_pipeline():
    """Run the complete training pipeline"""
    # Setup logging
    log_file = setup_logging()
    
    # Create directories
    create_directories()
    
    # Check GPU
    check_gpu_availability()
    
    try:
        logger.info("="*70)
        logger.info("STARTING OCR TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Dataset: {DATASET_PATH}")
        logger.info(f"Output: {OUTPUT_DIR}")
        logger.info(f"Model: {MODEL_DIR}")
        
        # Step 1: Download dataset
        dataset_path = step1_download()
        
        # Step 2: Split data
        splits_dir = step2_split(dataset_path)
        
        # Step 3: Prepare crops
        crops_dir = step3_prepare_crops(dataset_path, splits_dir)
        
        # Step 4: Generate lstmf files
        lstmf_dir = step4_generate_lstmf(crops_dir)
        
        # Step 5: Train model
        model_path = step5_train(lstmf_dir)
        
        # Step 6: Evaluate
        step6_evaluate(crops_dir, model_path)
        
        logger.info("\n" + "="*70)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        logger.info("="*70)
        logger.info(f"\n✅ Model saved: {model_path}")
        logger.info(f"✅ Charts saved: {CHARTS_DIR}")
        logger.info(f"✅ Logs saved: {log_file}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        logger.error("="*70)
        raise


if __name__ == "__main__":
    run_full_pipeline()