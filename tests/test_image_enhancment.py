# tests/simple_enhancement_test.py
import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from ocr.image_enhanced import ImageEnhancer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set directories
INPUT_DIR = Path("/home/alina/DiplomaWork/ocr_telegram_bot/tests/test_images")
OUTPUT_DIR = Path("/home/alina/DiplomaWork/ocr_telegram_bot/tests/enhanced_results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("IMAGE ENHANCEMENT BATCH PROCESS")
print("="*60)

# Find all images in input directory
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
images = []

for ext in image_extensions:
    images.extend(INPUT_DIR.glob(f"*{ext}"))
    images.extend(INPUT_DIR.glob(f"*{ext.upper()}"))

if not images:
    print(f"No images found in {INPUT_DIR}")
    print("Please add images to test_images/ folder")
    sys.exit(1)

print(f"Found {len(images)} image(s)")
print(f"Output directory: {OUTPUT_DIR}\n")

# Create enhancer
enhancer = ImageEnhancer()

# Process each image
for img_path in images:
    print(f"Processing: {img_path.name}")
    
    # Load image
    img = Image.open(img_path)
    
    # Apply enhancement steps
    step1 = enhancer.to_grayscale(img)
    step2 = enhancer.upscale(step1, min_height=64)
    step3 = enhancer.remove_noise(step2, strength=3)
    step4 = enhancer.auto_contrast(step3)
    step5 = enhancer.sharpen(step4, factor=1.5)
    step6 = enhancer.deskew(step5)
    step7 = enhancer.otsu_threshold(step6)
    
    # Create collage image
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    steps = [
        ("Original", img),
        ("Grayscale", step1),
        ("Upscaled", step2),
        ("Noise removed", step3),
        ("Auto contrast", step4),
        ("Sharpened", step5),
        ("Deskewed", step6),
        ("Final B&W", step7)
    ]
    
    for i, (title, image) in enumerate(steps):
        if image.mode == 'RGB':
            axes[i].imshow(image)
        else:
            axes[i].imshow(image, cmap='gray')
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Enhancement Steps - {img_path.name}', fontsize=12)
    plt.tight_layout()
    
    # Save collage
    collage_file = OUTPUT_DIR / f"{img_path.stem}_steps.png"
    plt.savefig(collage_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    original_arr = np.array(step1)
    final_arr = np.array(step7)
    
    original_std = original_arr.std()
    final_std = final_arr.std()
    improvement = ((final_std - original_std) / original_std * 100) if original_std > 0 else 0
    
    # Save statistics
    stats_file = OUTPUT_DIR / f"{img_path.stem}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"IMAGE: {img_path.name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Original size: {img.size[0]}x{img.size[1]}\n")
        f.write(f"Enhanced size: {step7.size[0]}x{step7.size[1]}\n\n")
        f.write("Contrast (Standard Deviation):\n")
        f.write("-"*30 + "\n")
        f.write(f"Original:  {original_std:.2f}\n")
        f.write(f"Enhanced:  {final_std:.2f}\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
    
    print(f"  ✓ Saved: {collage_file.name}")
    print(f"  ✓ Saved: {stats_file.name}")
    print()

print("="*60)
print("COMPLETED!")
print("="*60)
print(f"\nAll results saved in: {OUTPUT_DIR}")