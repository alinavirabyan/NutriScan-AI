"""
OCR Comparison Module
=====================
Compare Tesseract (base and fine-tuned) with EasyOCR.
Provides metrics, visualizations, and performance analysis.

Usage:
    from ocr.compare_tesseract_ocr import OCRComparator
    comparator = OCRComparator()
    results = comparator.compare_on_image("image.jpg")
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
import logging

import pytesseract
import easyocr

# Import your enhancement module
from ocr.image_enhanced import ImageEnhancer

logger = logging.getLogger(__name__)


class OCRComparator:
    """
    Compare different OCR engines on the same image.
    Supports Tesseract (base + fine-tuned) and EasyOCR.
    """
    
    def __init__(self, tessdata_dir="/usr/share/tesseract-ocr/5/tessdata", 
                 fine_tuned_model="eng_textocr",
                 easyocr_langs=['en'],
                 use_gpu=False):
        """
        Initialize OCR comparator.
        
        Args:
            tessdata_dir: Path to Tesseract tessdata directory
            fine_tuned_model: Name of fine-tuned model
            easyocr_langs: Languages for EasyOCR
            use_gpu: Use GPU for EasyOCR if available
        """
        self.tessdata_dir = tessdata_dir
        self.fine_tuned_model = fine_tuned_model
        self.easyocr_langs = easyocr_langs
        self.use_gpu = use_gpu
        
        # Initialize EasyOCR reader (lazy loading)
        self._easyocr_reader = None
        
        # Initialize image enhancer
        self.enhancer = ImageEnhancer()
        
        logger.info(f"OCR Comparator initialized")
        logger.info(f"  Fine-tuned model: {fine_tuned_model}")
        logger.info(f"  EasyOCR languages: {easyocr_langs}")
    
    @property
    def easyocr_reader(self):
        """Lazy load EasyOCR reader"""
        if self._easyocr_reader is None:
            logger.info("Loading EasyOCR model (first run may take ~30 seconds)...")
            self._easyocr_reader = easyocr.Reader(
                self.easyocr_langs, 
                gpu=self.use_gpu
            )
        return self._easyocr_reader
    
    def run_tesseract_base(self, image) -> dict:
        """
        Run base Tesseract (eng) on image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            dict with text, time, engine name
        """
        start = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Run Tesseract
        config = "--psm 6"  # Uniform block of text
        text = pytesseract.image_to_string(img, lang="eng", config=config).strip()
        
        elapsed = time.time() - start
        
        return {
            "engine": "Tesseract (Base)",
            "text": text,
            "time": elapsed,
            "confidence": self._estimate_confidence(text)
        }
    
    def run_tesseract_finetuned(self, image) -> dict:
        """
        Run fine-tuned Tesseract model on image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            dict with text, time, engine name
        """
        start = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Run fine-tuned Tesseract
        config = f"--tessdata-dir {self.tessdata_dir} --psm 6"
        
        try:
            text = pytesseract.image_to_string(
                img, 
                lang=self.fine_tuned_model, 
                config=config
            ).strip()
        except Exception as e:
            text = f"[ERROR: Fine-tuned model '{self.fine_tuned_model}' not found. Install with: sudo cp models/tesseract/{self.fine_tuned_model}.traineddata {self.tessdata_dir}/]"
            logger.error(f"Fine-tuned model error: {e}")
        
        elapsed = time.time() - start
        
        return {
            "engine": "Tesseract (Fine-tuned)",
            "text": text,
            "time": elapsed,
            "confidence": self._estimate_confidence(text)
        }
    
    def run_easyocr(self, image) -> dict:
        """
        Run EasyOCR on image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            dict with text, time, engine name, detections
        """
        start = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        else:
            img = image
            img_array = np.array(img)
        
        # Run EasyOCR
        result = self.easyocr_reader.readtext(img_array)
        
        # Combine all detected text
        texts = []
        detections = []
        for (bbox, text, confidence) in result:
            texts.append(text)
            detections.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        full_text = " ".join(texts)
        avg_confidence = sum([d['confidence'] for d in detections]) / len(detections) if detections else 0
        
        elapsed = time.time() - start
        
        return {
            "engine": "EasyOCR",
            "text": full_text,
            "time": elapsed,
            "confidence": avg_confidence,
            "detections": detections
        }
    
    def _estimate_confidence(self, text):
        """
        Simple confidence estimation based on text length and content.
        (Tesseract doesn't provide direct confidence for all modes)
        """
        if not text:
            return 0.0
        # Rough heuristic: longer text with more words = higher confidence
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Simple score based on words length
        base_score = min(len(text) / 100, 0.95)
        return base_score
    
    def compare_on_image(self, image_path, ground_truth=None, preprocess=True):
        """
        Run all OCR engines on a single image and compare results.
        
        Args:
            image_path: Path to image file
            ground_truth: Optional ground truth text for metrics
            preprocess: Apply image enhancement before OCR
            
        Returns:
            dict with results from all engines
        """
        print("\n" + "="*60)
        print("OCR ENGINE COMPARISON")
        print("="*60)
        print(f"Image: {image_path}")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Load original image
        original = Image.open(image_path).convert("RGB")
        print(f"Size: {original.size[0]}x{original.size[1]}")
        
        # Apply preprocessing
        if preprocess:
            print("\nApplying image enhancement...")
            enhanced_img, _ = self.enhancer.enhance_image(image_path)
            # For EasyOCR, use color image
            easyocr_img = original
            # For Tesseract, use preprocessed grayscale
            tesseract_img = enhanced_img
            print("  Enhancement completed")
        else:
            tesseract_img = original.convert('L')
            easyocr_img = original
        
        print("\nRunning OCR engines...")
        print("-" * 40)
        
        # Run all engines
        results = []
        
        # Tesseract Base
        print("  Running Tesseract (Base)...")
        r1 = self.run_tesseract_base(tesseract_img)
        results.append(r1)
        print(f"    Time: {r1['time']:.2f}s")
        
        # Tesseract Fine-tuned
        print("  Running Tesseract (Fine-tuned)...")
        r2 = self.run_tesseract_finetuned(tesseract_img)
        results.append(r2)
        print(f"    Time: {r2['time']:.2f}s")
        
        # EasyOCR
        print("  Running EasyOCR...")
        r3 = self.run_easyocr(easyocr_img)
        results.append(r3)
        print(f"    Time: {r3['time']:.2f}s")
        
        # Print results
        self._print_results(results, ground_truth)
        
        # Create visualizations
        output_dir = Path(image_path).parent / "ocr_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison chart
        self._create_comparison_chart(results, ground_truth, output_dir)
        
        # Create EasyOCR detection visualization
        if 'detections' in r3:
            self._visualize_easyocr_detections(
                original, 
                r3['detections'], 
                output_dir / "easyocr_detections.png"
            )
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            "tesseract_base": r1,
            "tesseract_finetuned": r2,
            "easyocr": r3
        }
    
    def _print_results(self, results, ground_truth=None):
        """Print comparison results"""
        print("\n" + "="*60)
        print("EXTRACTED TEXT")
        print("="*60)
        
        for r in results:
            print(f"\n[{r['engine']}]")
            print(f"Time: {r['time']:.2f}s")
            print(f"Confidence: {r['confidence']:.1%}")
            print(f"Text: {r['text'][:200]}")
        
        if ground_truth:
            print("\n" + "="*60)
            print("METRICS (lower is better)")
            print("="*60)
            print(f"{'Engine':<25} {'CER':>8} {'WER':>8}")
            print("-" * 42)
            
            for r in results:
                cer = self._compute_cer(ground_truth, r['text'])
                wer = self._compute_wer(ground_truth, r['text'])
                print(f"{r['engine']:<25} {cer:>8.3f} {wer:>8.3f}")
    
    def _compute_cer(self, gt, pred):
        """Character Error Rate"""
        if not gt:
            return 0.0
        return self._levenshtein(gt.lower(), pred.lower()) / len(gt)
    
    def _compute_wer(self, gt, pred):
        """Word Error Rate"""
        gt_words = gt.lower().split()
        pred_words = pred.lower().split()
        if not gt_words:
            return 0.0
        return self._levenshtein(gt_words, pred_words) / len(gt_words)
    
    def _levenshtein(self, a, b):
        """Levenshtein distance"""
        if isinstance(a, str):
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
    
    def _create_comparison_chart(self, results, ground_truth, output_dir):
        """Create bar chart comparing OCR engines"""
        
        engines = [r['engine'] for r in results]
        times = [r['time'] for r in results]
        colors = ['#EF5350', '#42A5F5', '#66BB6A']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time chart
        ax1 = axes[0]
        bars = ax1.bar(engines, times, color=colors, edgecolor='white')
        for bar, val in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01,
                    f'{val:.2f}s', ha='center', fontweight='bold')
        ax1.set_title('Processing Time (lower = faster)', fontweight='bold')
        ax1.set_ylabel('Seconds')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Confidence chart
        ax2 = axes[1]
        confidences = [r['confidence'] for r in results]
        bars = ax2.bar(engines, confidences, color=colors, edgecolor='white')
        for bar, val in zip(bars, confidences):
            ax2.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', fontweight='bold')
        ax2.set_title('Confidence Score (higher = better)', fontweight='bold')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('OCR Engine Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        chart_path = output_dir / "ocr_comparison_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart saved: {chart_path}")
    
    def _visualize_easyocr_detections(self, image, detections, save_path):
        """Draw bounding boxes for EasyOCR detections"""
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        for det in detections:
            bbox = det['bbox']
            text = det['text']
            confidence = det['confidence']
            
            # Convert bbox to points
            points = [(int(p[0]), int(p[1])) for p in bbox]
            
            # Draw polygon
            draw.polygon(points, outline=(0, 200, 100), width=2)
            
            # Draw text label
            draw.text(
                (points[0][0], points[0][1] - 20),
                f"{text} ({confidence:.0%})",
                fill=(0, 200, 100)
            )
        
        img_draw.save(save_path)
        print(f"  Detection visualization saved: {save_path}")
        return img_draw
    
    def compare_multiple_images(self, image_paths, ground_truths=None):
        """
        Compare OCR engines on multiple images.
        
        Args:
            image_paths: List of image paths
            ground_truths: List of ground truth texts (optional)
        """
        all_results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_path}")
            
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            result = self.compare_on_image(img_path, ground_truth=gt, preprocess=True)
            all_results.append(result)
        
        return all_results


# For backward compatibility
def compare_ocr_engines(image_path, ground_truth=None, preprocess=True):
    """Wrapper function for backward compatibility"""
    comparator = OCRComparator()
    return comparator.compare_on_image(image_path, ground_truth, preprocess)


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 compare_tesseract_ocr.py <image_path> [ground_truth]")
        print("Example: python3 compare_tesseract_ocr.py spring.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    ground_truth = sys.argv[2] if len(sys.argv) > 2 else None
    
    comparator = OCRComparator()
    results = comparator.compare_on_image(image_path, ground_truth)