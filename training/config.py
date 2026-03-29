# training/config.py
"""
Configuration for OCR Training Pipeline
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# DATASET PATHS
# ============================================================
# Your dataset location - UPDATE THIS TO YOUR ACTUAL PATH
DATASET_PATH = PROJECT_ROOT / "datasets" / "textocr"

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"
LSTMF_DIR = OUTPUT_DIR / "lstmf_files"
SPLITS_DIR = OUTPUT_DIR / "splits"
LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ============================================================
# MODEL PATHS
# ============================================================
MODEL_DIR = PROJECT_ROOT / "models" / "tesseract"
FINAL_MODEL_PATH = MODEL_DIR / "eng_textocr.traineddata"

# ============================================================
# TESSERACT PATHS
# ============================================================
TESSDATA_DIR = "/usr/share/tesseract-ocr/5/tessdata"
SYSTEM_TESSDATA = "/usr/share/tesseract-ocr/5/tessdata"
BASE_LANG = "eng_best"  # Base model to fine-tune from
MODEL_NAME = "eng_textocr"  # Name of your fine-tuned model

# ============================================================
# TRAINING PARAMETERS
# ============================================================
# Set to None to use all samples, or a number for testing
MAX_SAMPLES = 5000  # Start with 5000 for testing, then set to None for full training

# Training iterations
MAX_ITERATIONS = 150000
LEARNING_RATE = 0.00003

# Data cleaning
MIN_TEXT_LEN = 3
PADDING = 4

# Data split
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42

# ============================================================
# CREATE DIRECTORIES
# ============================================================
def create_directories():
    """Create all necessary directories"""
    directories = [
        OUTPUT_DIR,
        CHARTS_DIR,
        TRAINING_DATA_DIR,
        LSTMF_DIR,
        SPLITS_DIR,
        LOGS_DIR,
        CHECKPOINT_DIR,
        MODEL_DIR,
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

# ============================================================
# CHECK PATHS
# ============================================================
def check_paths():
    """Check if all required paths exist"""
    issues = []
    
    if not DATASET_PATH.exists():
        issues.append(f"Dataset not found: {DATASET_PATH}")
    
    if not (DATASET_PATH / "img.csv").exists():
        issues.append(f"Missing img.csv in: {DATASET_PATH}")
    
    if not (DATASET_PATH / "annot.csv").exists():
        issues.append(f"Missing annot.csv in: {DATASET_PATH}")
    
    if issues:
        print("\n⚠️ WARNINGS:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease update DATASET_PATH in config.py to your dataset location")
        return False
    
    print("✅ All paths are configured correctly!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("CONFIGURATION CHECK")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Model dir: {MODEL_DIR}")
    print("-"*60)
    
    create_directories()
    check_paths()