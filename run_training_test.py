import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent))

print("="*70)
print("TRAINING PIPELINE TEST")
print("="*70)

# Test config
print("\n1. Testing configuration...")
try:
    from training.config import *
    print("   ✅ Config imported")
    print(f"   Dataset path: {DATASET_PATH}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Model dir: {MODEL_DIR}")
except Exception as e:
    print(f"   ❌ Config error: {e}")

# Test utils
print("\n2. Testing utilities...")
try:
    from training.utils import *
    print("   ✅ Utils imported")
    print(f"   Levenshtein test: {levenshtein('test', 'test')}")
except Exception as e:
    print(f"   ❌ Utils error: {e}")

# Test visualization
print("\n3. Testing visualization...")
try:
    from training.visualization import *
    print("   ✅ Visualization imported")
except Exception as e:
    print(f"   ❌ Visualization error: {e}")

# Check dataset
print("\n4. Checking dataset...")
if DATASET_PATH.exists():
    print(f"   ✅ Dataset found at {DATASET_PATH}")
    
    # Check required files
    required = ["img.csv", "annot.csv"]
    for f in required:
        if (DATASET_PATH / f).exists():
            print(f"   ✅ {f} found")
        else:
            print(f"   ❌ {f} missing")
else:
    print(f"   ⚠️ Dataset not found at {DATASET_PATH}")
    print("   To download: run training pipeline or download manually")

# Check Tesseract
print("\n5. Checking Tesseract...")
try:
    import pytesseract
    print("   ✅ pytesseract installed")
    
    import subprocess
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
    version = result.stdout.split()[1] if result.stdout else "unknown"
    print(f"   ✅ Tesseract version: {version}")
except Exception as e:
    print(f"   ❌ Tesseract error: {e}")

print("\n" + "="*70)
print("✅ All modules tested!")
print("="*70)
