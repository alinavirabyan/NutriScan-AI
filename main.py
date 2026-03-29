#!/usr/bin/env python3
"""
main.py — Entry point for OCR Telegram Bot project

Usage:
    python3 main.py train       # Run full training pipeline
    python3 main.py evaluate    # Evaluate existing model
    python3 main.py bot         # Start Telegram bot
    python3 main.py pipeline    # Run OCR pipeline on an image
    python3 main.py check       # Check configuration
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def cmd_train():
    """Run full Tesseract fine-tuning pipeline."""
    from training.train_pipeline import run_full_pipeline
    run_full_pipeline()


def cmd_evaluate():
    """Evaluate the trained model on test data."""
    from training.config import FINAL_MODEL_PATH, TRAINING_DATA_DIR
    from training.evaluation import evaluate_on_crops, evaluate_on_test_split

    model = str(FINAL_MODEL_PATH)
    if not FINAL_MODEL_PATH.exists():
        log.error(f"Model not found: {model}")
        log.info("Run: python3 main.py train")
        sys.exit(1)

    print("\n=== Evaluation on training crops ===")
    evaluate_on_crops(str(TRAINING_DATA_DIR), model)

    print("\n=== Evaluation on TEST split (honest score) ===")
    evaluate_on_test_split(model)


def cmd_bot():
    """Start the Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set!")
        log.info("Run: export TELEGRAM_BOT_TOKEN='your_token'")
        log.info("Get token from @BotFather on Telegram")
        sys.exit(1)

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        log.warning("MISTRAL_API_KEY not set — summarization will be disabled")
        log.info("Get free key from: https://console.mistral.ai/")

    from telegram_bot.bot import run_bot
    run_bot(token)


def cmd_pipeline(image_path: str = None):
    """Run OCR pipeline on an image."""
    if not image_path:
        log.error("Please provide an image path")
        log.info("Usage: python3 main.py pipeline /path/to/image.jpg")
        sys.exit(1)

    if not os.path.exists(image_path):
        log.error(f"Image not found: {image_path}")
        sys.exit(1)

    from ocr.pipeline import run_pipeline
    result = run_pipeline(image_path, verbose=True)

    # Save result
    out_file = os.path.splitext(image_path)[0] + "_result.txt"
    with open(out_file, "w") as f:
        f.write(f"Image: {image_path}\n\n")
        f.write(f"Extracted text:\n{result.get('combined_text', '')}\n\n")
        f.write(f"Summary:\n{result.get('summary', '')}\n")
    log.info(f"Result saved: {out_file}")


def cmd_check():
    """Check configuration and dependencies."""
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)

    # Config
    print("\n1. Configuration...")
    try:
        from training.config import (
            DATASET_PATH, OUTPUT_DIR, MODEL_DIR,
            TESSDATA_DIR, BASE_LANG, MODEL_NAME,
            MAX_SAMPLES, MAX_ITERATIONS
        )
        print(f"   ✅ Config loaded")
        print(f"   Dataset    : {DATASET_PATH}")
        print(f"   Output     : {OUTPUT_DIR}")
        print(f"   Model      : {MODEL_DIR}")
        print(f"   Samples    : {MAX_SAMPLES}")
        print(f"   Iterations : {MAX_ITERATIONS}")
    except Exception as e:
        print(f"   ❌ Config error: {e}")

    # Dataset
    print("\n2. Dataset...")
    try:
        from training.config import DATASET_PATH
        if DATASET_PATH.exists():
            has_img   = (DATASET_PATH / "img.csv").exists()
            has_annot = (DATASET_PATH / "annot.csv").exists()
            print(f"   ✅ Dataset folder found")
            print(f"   {'✅' if has_img   else '❌'} img.csv")
            print(f"   {'✅' if has_annot else '❌'} annot.csv")
        else:
            print(f"   ⚠️  Dataset not found at {DATASET_PATH}")
            print(f"   Run: python3 main.py train (downloads automatically)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Tesseract
    print("\n3. Tesseract...")
    try:
        import subprocess
        r = subprocess.run(["tesseract", "--version"],
                           capture_output=True, text=True)
        version = r.stdout.split()[1] if r.stdout else "unknown"
        print(f"   ✅ Tesseract {version}")

        import os
        from training.config import TESSDATA_DIR
        eng_best = os.path.join(TESSDATA_DIR, "eng_best.traineddata")
        print(f"   {'✅' if os.path.exists(eng_best) else '❌'} eng_best.traineddata")

        eng_ft = os.path.join(TESSDATA_DIR, "eng_textocr.traineddata")
        print(f"   {'✅' if os.path.exists(eng_ft) else '⚠️ '} eng_textocr.traineddata")
    except Exception as e:
        print(f"   ❌ Tesseract error: {e}")

    # Python packages
    print("\n4. Python packages...")
    packages = [
        "pytesseract", "easyocr", "PIL", "cv2",
        "pandas", "numpy", "matplotlib", "sklearn",
        "kagglehub", "mistralai", "telegram"
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} — pip install {pkg}")

    # Environment variables
    print("\n5. Environment variables...")
    bot_token    = os.getenv("TELEGRAM_BOT_TOKEN")
    mistral_key  = os.getenv("MISTRAL_API_KEY")
    print(f"   {'✅' if bot_token   else '❌'} TELEGRAM_BOT_TOKEN")
    print(f"   {'✅' if mistral_key else '❌'} MISTRAL_API_KEY")

    print("\n" + "="*60)
    print("Check complete!")
    print("="*60)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

COMMANDS = {
    "train":    cmd_train,
    "evaluate": cmd_evaluate,
    "bot":      cmd_bot,
    "check":    cmd_check,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "pipeline":
        image = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_pipeline(image)
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)