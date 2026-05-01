#!/usr/bin/env python3
"""
CLEAN PIPELINE - Minimal output
Shows: dataset info, progress every 50K, final directory contents
"""

import ast, subprocess, sys, time, shutil, random, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm


FINAL_WORK = Path("/home/alina/DiplomaWork/Final_Work")
DATASET_PATH = Path("/home/alina/DiplomaWork/Diploma_work/datasets/textocr")

OUTPUTS = FINAL_WORK / "outputs"
ALL_DATA = OUTPUTS / "all_data"
TRAIN_LSTMF = OUTPUTS / "train_lstmf"
TEST_LSTMF = OUTPUTS / "test_lstmf"
TRAIN_LIST = OUTPUTS / "train_list_lstmf"
CHECKPOINTS = OUTPUTS / "checkpoints"
EVAL_RESULTS = OUTPUTS / "evaluation_results"

for d in [ALL_DATA, TRAIN_LSTMF, TEST_LSTMF, CHECKPOINTS, EVAL_RESULTS]:
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
TRAIN_LIST.mkdir(parents=True, exist_ok=True)

PADDING, MIN_HEIGHT = 12, 20
MAX_SAMPLES = None  # None = ALL data
TEST_RATIO = 0.20
RESUME_FILE = ALL_DATA / "progress.json"

print("=" * 60)
print("  Dataset:", DATASET_PATH)
print("  Samples:", "ALL" if MAX_SAMPLES is None else f"{MAX_SAMPLES:,}")

img_df = pd.read_csv(DATASET_PATH / "img.csv", index_col=0)
annot_df = pd.read_csv(DATASET_PATH / "annot.csv", index_col=0)

for c in [DATASET_PATH / "train_val_images" / "train_images", DATASET_PATH / "train_images"]:
    if c.exists(): images_folder = c; break

print(f"  Images: {len(img_df):,} | Annotations: {len(annot_df):,}")

def is_good(text):
    """Check if annotation text is valid for training.

    Filters out text that is too short, too long, has no letters,
    too many digits, or too many non-alphanumeric characters.

    Args:
        text: The annotation text to validate.

    Returns:
        True if the text passes all quality checks.
    """
    if not isinstance(text, str): return False
    text = text.strip()
    if len(text) < 3 or len(text) > 25: return False
    if not any(c.isalpha() for c in text): return False
    if sum(c.isdigit() for c in text) > len(text) * 0.5: return False
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and alnum / len(text) < 0.6: return False
    return True

annot_df = annot_df[annot_df['utf8_string'].notna()]
annot_df = annot_df[annot_df['utf8_string'].astype(str).apply(is_good)]
print(f"  Valid: {len(annot_df):,}")

if MAX_SAMPLES and len(annot_df) > MAX_SAMPLES:
    annot_df = annot_df.sample(n=MAX_SAMPLES, random_state=42)
    print(f"  Selected: {len(annot_df):,}")

print("=" * 60)

start_idx = 0
if RESUME_FILE.exists():
    start_idx = json.loads(RESUME_FILE.read_text()).get("last_idx", 0)
    print(f"🔄 Resuming from {start_idx:,}")


id_to_file = dict(zip(img_df['id'], img_df['file_name']))
img_cache = {}
saved, lstmf_ok = start_idx, 0
t0 = time.time()
rows = annot_df.iloc[start_idx:]

for idx, (_, row) in enumerate(tqdm(rows.iterrows(), total=len(rows), desc="Processing")):
    actual_idx = start_idx + idx
    try:
        image_id = row['image_id']
        text = str(row['utf8_string']).strip()
        bbox = ast.literal_eval(row['bbox']) if isinstance(row['bbox'], str) else list(row['bbox'])
        
        fname = id_to_file.get(image_id)
        if not fname: continue
        img_path = images_folder / Path(fname).name
        if not img_path.exists(): continue
        
        if image_id not in img_cache:
            if len(img_cache) > 30: img_cache.clear()
            img_cache[image_id] = Image.open(img_path).convert("RGB")
        img = img_cache[image_id]
        
        x, y, w, h = [int(float(v)) for v in bbox]
        x1, y1 = max(0, x-PADDING), max(0, y-PADDING)
        x2, y2 = min(img.width, x+w+PADDING), min(img.height, y+h+PADDING)
        if x2-x1 < 4 or y2-y1 < 4: continue
        
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.convert("L")
        crop = ImageOps.autocontrast(crop, cutoff=2)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
        
        if crop.height < MIN_HEIGHT:
            scale = MIN_HEIGHT / crop.height
            crop = crop.resize((max(1, int(crop.width*scale)), MIN_HEIGHT), Image.Resampling.LANCZOS)
        if crop.width < MIN_HEIGHT:
            crop = crop.resize((MIN_HEIGHT, crop.height), Image.Resampling.LANCZOS)
        
        stem = f"word_{actual_idx:07d}"
        png_path = ALL_DATA / f"{stem}.png"
        crop.save(png_path, format="PNG")
        (ALL_DATA / f"{stem}.txt").write_text(text, encoding="utf-8")
        w_img, h_img = crop.size
        (ALL_DATA / f"{stem}.box").write_text(
            f"WordStr 0 0 {w_img} {h_img} 0 #{text}\n\t 0 0 {w_img} {h_img} 0\n", encoding="utf-8")
        
        out_stem = ALL_DATA / stem
        subprocess.run(["tesseract", str(png_path), str(out_stem), "--psm", "7", "-l", "eng", "lstm.train"],
                       capture_output=True, text=True)
        if out_stem.with_suffix(".lstmf").exists(): lstmf_ok += 1
        saved += 1
        
        # ── REPORT EVERY 50,000 ──
        if (actual_idx + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (saved - start_idx) / elapsed
            remaining = (len(annot_df) - actual_idx) / rate if rate > 0 else 0
            print(f"\n  📊 {actual_idx+1:,}/{len(annot_df):,} | LSTM: {lstmf_ok:,} | "
                  f"Elapsed: {elapsed/60:.0f}min | ETA: {remaining/60:.0f}min\n")
            RESUME_FILE.write_text(json.dumps({"last_idx": actual_idx + 1}))
        
    except Exception:
        continue
    
    if actual_idx % 5000 == 0:
        RESUME_FILE.write_text(json.dumps({"last_idx": actual_idx}))

RESUME_FILE.write_text(json.dumps({"last_idx": len(annot_df), "done": True}))
elapsed = time.time() - t0

print(f"\n📊 Splitting train/test...")
lstmf_files = sorted(ALL_DATA.glob("word_*.lstmf"))
random.seed(42)
random.shuffle(lstmf_files)
split_idx = int(len(lstmf_files) * (1 - TEST_RATIO))
train_files, test_files = lstmf_files[:split_idx], lstmf_files[split_idx:]

for lstmf in train_files:
    shutil.copy2(lstmf, TRAIN_LSTMF / lstmf.name)
for lstmf in test_files:
    shutil.copy2(lstmf, TEST_LSTMF / lstmf.name)
    for ext in [".png", ".txt"]:
        f = lstmf.with_suffix(ext)
        if f.exists(): shutil.copy2(f, TEST_LSTMF / f.name)

# all.txt
all_txt = TRAIN_LSTMF / "all.txt"
with open(all_txt, 'w') as f:
    for lstmf in sorted(TRAIN_LSTMF.glob("*.lstmf")): f.write(f"{lstmf}\n")

# train_list.txt
train_list_txt = TRAIN_LIST / "train_list.txt"
with open(train_list_txt, 'w') as f:
    for lstmf in sorted(TRAIN_LSTMF.glob("*.lstmf")): f.write(f"{lstmf}\n")



print(f"\n{'='*60}")
print(f"  ✅ DONE! ({elapsed/3600:.1f} hours)")
print(f"{'='*60}")
print(f"  all_data/       → {saved:,} PNG + TXT + BOX | {lstmf_ok:,} LSTM")
print(f"  train_lstmf/    → {len(train_files):,} LSTM files + all.txt")
print(f"  test_lstmf/     → {len(test_files):,} LSTM + PNG + TXT")
print(f"  train_list_lstmf/ → train_list.txt ({len(train_files):,} paths)")
print(f"  checkpoints/    → (empty)")
print(f"  evaluation_results/ → (empty)")
print(f"{'='*60}")