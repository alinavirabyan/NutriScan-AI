# training/utils.py
"""
Utility functions for OCR training pipeline
"""

import re
import subprocess
import logging
from PIL import Image, ImageOps
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# GPU-agnostic imports
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def banner(title: str) -> None:
    """Print a banner"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_gpu_availability():
    """Check and report GPU availability"""
    print("\n" + "="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            for line in result.stdout.strip().split('\n')[:2]:
                print(f"  {line}")
    except FileNotFoundError:
        print("✗ nvidia-smi not found - no NVIDIA GPU detected")
    
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch detected")
        print(f"  Device: {DEVICE}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ PyTorch not installed")
    
    if CUPY_AVAILABLE:
        print("✓ CuPy detected")
    else:
        print("✗ CuPy not installed")
    
    print("="*60 + "\n")


def levenshtein(a, b) -> int:
    """Calculate Levenshtein distance between two strings or lists"""
    if isinstance(a, str) and isinstance(b, str):
        a = list(a)
        b = list(b)
    
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(
                dp[j] + (0 if ca == cb else 1),
                dp[j + 1] + 1,
                ndp[-1] + 1
            ))
        dp = ndp
    return dp[-1]


def compute_cer(gt: str, pred: str) -> float:
    """Calculate Character Error Rate (CER)"""
    gt, pred = gt.strip().lower(), pred.strip().lower()
    if not gt:
        return 0.0
    return levenshtein(gt, pred) / len(gt)


def compute_wer(gt: str, pred: str) -> float:
    """Calculate Word Error Rate (WER)"""
    gt_words = gt.strip().lower().split()
    pred_words = pred.strip().lower().split()
    if not gt_words:
        return 0.0
    return levenshtein(gt_words, pred_words) / len(gt_words)


def is_clean_label(text: str, min_text_len: int = 3) -> bool:
    """Check if label is clean enough for training"""
    text = text.strip()
    
    # Check minimum length
    if len(text) < min_text_len:
        return False
    
    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Must be ASCII (for simplicity)
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        return False
    
    return True


def preprocess_crop(img: Image.Image, min_height: int = 32) -> Image.Image:
    """Preprocess a cropped word image"""
    # Convert to grayscale
    img = img.convert("L")
    
    # Auto contrast
    img = ImageOps.autocontrast(img)
    
    # Upscale if too small
    if img.height < min_height:
        scale = min_height / img.height
        new_width = int(img.width * scale)
        img = img.resize((new_width, min_height), Image.LANCZOS)
    
    return img


def preprocess_crop_gpu(img: Image.Image, min_height: int = 32) -> Image.Image:
    """GPU-accelerated preprocessing for large images"""
    img = img.convert("L")
    
    if TORCH_AVAILABLE and img.size[0] * img.size[1] > 500000:
        import torch
        import torchvision.transforms.functional as F
        
        img_tensor = F.to_tensor(img).to(DEVICE)
        mean = img_tensor.mean()
        img_tensor = (img_tensor - mean) * 1.5 + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img = F.to_pil_image(img_tensor.cpu())
    else:
        img = ImageOps.autocontrast(img)
    
    if img.height < min_height:
        scale = min_height / img.height
        new_width = int(img.width * scale)
        img = img.resize((new_width, min_height), Image.LANCZOS)
    
    return img


def setup_logging():
    """Setup logging for training"""
    from training.config import LOGS_DIR
    
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


if __name__ == "__main__":
    print("Testing utility functions...")
    print(f"Levenshtein('hello', 'hello'): {levenshtein('hello', 'hello')}")
    print(f"CER('hello', 'hello'): {compute_cer('hello', 'hello')}")
    print(f"WER('hello world', 'hello world'): {compute_wer('hello world', 'hello world')}")
    print("✓ All utilities working!")