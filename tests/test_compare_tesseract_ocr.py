# tests/full_ocr_comparison.py
"""
Complete OCR Comparison Pipeline:
1. Take one image
2. Apply all 7 enhancement steps
3. For EACH enhanced image, run Tesseract and EasyOCR
4. Compare metrics and find best OCR for each enhancement
5. Determine overall best combination
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ocr.image_enhanced import ImageEnhancer
from PIL import Image
import pytesseract
import easyocr


class FullOCRComparison:
    """Compare Tesseract and EasyOCR on all enhancement steps"""
    
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.easyocr_reader = None
        self.results = []
    
    def _init_easyocr(self):
        """Initialize EasyOCR"""
        if self.easyocr_reader is None:
            print("  Loading EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    def _run_tesseract(self, image):
        """Run Tesseract on image"""
        start = time.time()
        
        if image.mode != 'L':
            img = image.convert('L')
        else:
            img = image
        
        config = "--psm 6"
        text = pytesseract.image_to_string(img, lang="eng", config=config).strip()
        
        elapsed = time.time() - start
        
        return {
            "text": text,
            "time": elapsed,
            "length": len(text),
            "words": len(text.split())
        }
    
    def _run_easyocr(self, image):
        """Run EasyOCR on image"""
        self._init_easyocr()
        
        start = time.time()
        
        img_array = np.array(image)
        results = self.easyocr_reader.readtext(img_array)
        
        elapsed = time.time() - start
        
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": full_text,
            "time": elapsed,
            "confidence": avg_confidence,
            "length": len(full_text),
            "words": len(full_text.split()),
            "detections": len(results)
        }
    
    def enhance_image_fully(self, image_path):
        """Apply all enhancement steps and return list of all versions"""
        
        print("\n" + "="*70)
        print("STEP 1: APPLYING ALL ENHANCEMENT STEPS")
        print("="*70)
        
        # Load original
        original = Image.open(image_path).convert("RGB")
        print(f"Original: {original.size}")
        
        # Apply each enhancement step
        versions = []
        names = []
        
        # Step 1: Original (for reference)
        versions.append(original)
        names.append("Original")
        
        # Step 2: Grayscale
        step1 = self.enhancer.to_grayscale(original)
        versions.append(step1)
        names.append("Grayscale")
        
        # Step 3: Upscaled
        step2 = self.enhancer.upscale(step1, min_height=64)
        versions.append(step2)
        names.append("Upscaled")
        
        # Step 4: Noise removed
        step3 = self.enhancer.remove_noise(step2, strength=3)
        versions.append(step3)
        names.append("Noise removed")
        
        # Step 5: Auto contrast
        step4 = self.enhancer.auto_contrast(step3)
        versions.append(step4)
        names.append("Auto contrast")
        
        # Step 6: Sharpened
        step5 = self.enhancer.sharpen(step4, factor=1.5)
        versions.append(step5)
        names.append("Sharpened")
        
        # Step 7: Deskewed
        step6 = self.enhancer.deskew(step5)
        versions.append(step6)
        names.append("Deskewed")
        
        # Step 8: Final B&W (Otsu threshold)
        step7 = self.enhancer.otsu_threshold(step6)
        versions.append(step7)
        names.append("Final B&W")
        
        print(f"\n✓ Created {len(versions)} image versions")
        
        return versions, names
    
    def run_ocr_on_all_versions(self, versions, names):
        """Run both OCR engines on all image versions"""
        
        print("\n" + "="*70)
        print("STEP 2: RUNNING OCR ON ALL ENHANCEMENT VERSIONS")
        print("="*70)
        
        results = []
        
        for i, (name, img) in enumerate(zip(names, versions)):
            print(f"\n📸 Version {i+1}/{len(versions)}: {name}")
            print("-" * 40)
            
            # Tesseract
            print("  Tesseract...", end=" ", flush=True)
            tesseract_result = self._run_tesseract(img)
            print(f"done ({tesseract_result['time']:.2f}s)")
            
            # EasyOCR
            print("  EasyOCR...", end=" ", flush=True)
            easyocr_result = self._run_easyocr(img)
            print(f"done ({easyocr_result['time']:.2f}s)")
            
            # Calculate contrast
            if img.mode != 'RGB':
                contrast = np.array(img).std()
            else:
                contrast = np.array(img.convert('L')).std()
            
            results.append({
                "enhancement": name,
                "contrast": contrast,
                "tesseract_text": tesseract_result["text"],
                "tesseract_time": tesseract_result["time"],
                "tesseract_length": tesseract_result["length"],
                "tesseract_words": tesseract_result["words"],
                "easyocr_text": easyocr_result["text"],
                "easyocr_time": easyocr_result["time"],
                "easyocr_length": easyocr_result["length"],
                "easyocr_words": easyocr_result["words"],
                "easyocr_confidence": easyocr_result["confidence"],
                "easyocr_detections": easyocr_result["detections"]
            })
        
        return results
    
    def compare_and_display(self, results):
        """Compare results and display metrics"""
        
        print("\n" + "="*70)
        print("STEP 3: COMPARISON METRICS")
        print("="*70)
        
        # Create comparison table
        table_data = []
        
        for r in results:
            # Determine which OCR is better for this enhancement
            if r["tesseract_length"] >= r["easyocr_length"]:
                better_ocr = "Tesseract"
                diff = r["tesseract_length"] - r["easyocr_length"]
            else:
                better_ocr = "EasyOCR"
                diff = r["easyocr_length"] - r["tesseract_length"]
            
            table_data.append([
                r["enhancement"],
                f"{r['contrast']:.1f}",
                f"{r['tesseract_time']:.2f}s",
                f"{r['tesseract_length']}",
                f"{r['easyocr_time']:.2f}s",
                f"{r['easyocr_length']}",
                f"{r['easyocr_confidence']:.1%}",
                better_ocr
            ])
        
        # Display table
        headers = [
            "Enhancement", "Contrast", "Tess Time", "Tess Len",
            "Easy Time", "Easy Len", "Easy Conf", "Better OCR"
        ]
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Find best enhancement for Tesseract
        best_tesseract = max(results, key=lambda x: x["tesseract_length"])
        
        # Find best enhancement for EasyOCR
        best_easyocr = max(results, key=lambda x: x["easyocr_length"])
        
        # Find overall best
        print("\n" + "="*70)
        print("STEP 4: BEST RESULTS")
        print("="*70)
        
        print(f"\n🏆 BEST FOR TESSERACT:")
        print(f"   Enhancement: {best_tesseract['enhancement']}")
        print(f"   Contrast: {best_tesseract['contrast']:.1f}")
        print(f"   Text length: {best_tesseract['tesseract_length']} characters")
        print(f"   Words: {best_tesseract['tesseract_words']}")
        print(f"   Time: {best_tesseract['tesseract_time']:.2f}s")
        print(f"   Text: {best_tesseract['tesseract_text'][:100]}...")
        
        print(f"\n🏆 BEST FOR EASYOCR:")
        print(f"   Enhancement: {best_easyocr['enhancement']}")
        print(f"   Contrast: {best_easyocr['contrast']:.1f}")
        print(f"   Text length: {best_easyocr['easyocr_length']} characters")
        print(f"   Words: {best_easyocr['easyocr_words']}")
        print(f"   Time: {best_easyocr['easyocr_time']:.2f}s")
        print(f"   Confidence: {best_easyocr['easyocr_confidence']:.1%}")
        print(f"   Text: {best_easyocr['easyocr_text'][:100]}...")
        
        # Compare original vs best enhancement
        original = results[0]
        best_overall = max(results[1:], key=lambda x: max(x["tesseract_length"], x["easyocr_length"]))
        
        print("\n" + "="*70)
        print("STEP 5: IMPROVEMENT ANALYSIS")
        print("="*70)
        
        tesseract_improvement = ((best_tesseract["tesseract_length"] - original["tesseract_length"]) / max(original["tesseract_length"], 1)) * 100
        easyocr_improvement = ((best_easyocr["easyocr_length"] - original["easyocr_length"]) / max(original["easyocr_length"], 1)) * 100
        
        print(f"\n📊 TESSERACT IMPROVEMENT:")
        print(f"   Original: {original['tesseract_length']} chars")
        print(f"   Best ({best_tesseract['enhancement']}): {best_tesseract['tesseract_length']} chars")
        print(f"   Improvement: +{tesseract_improvement:.1f}%")
        
        print(f"\n📊 EASYOCR IMPROVEMENT:")
        print(f"   Original: {original['easyocr_length']} chars")
        print(f"   Best ({best_easyocr['enhancement']}): {best_easyocr['easyocr_length']} chars")
        print(f"   Improvement: +{easyocr_improvement:.1f}%")
        
        # Final recommendation
        print("\n" + "="*70)
        print("FINAL RECOMMENDATION")
        print("="*70)
        
        # Compare best Tesseract vs best EasyOCR
        if best_tesseract["tesseract_length"] > best_easyocr["easyocr_length"]:
            print(f"\n✅ BEST OCR ENGINE: Tesseract")
            print(f"   Best enhancement: {best_tesseract['enhancement']}")
            print(f"   Text length: {best_tesseract['tesseract_length']} characters")
            print(f"   Processing time: {best_tesseract['tesseract_time']:.2f}s")
        elif best_easyocr["easyocr_length"] > best_tesseract["tesseract_length"]:
            print(f"\n✅ BEST OCR ENGINE: EasyOCR")
            print(f"   Best enhancement: {best_easyocr['enhancement']}")
            print(f"   Text length: {best_easyocr['easyocr_length']} characters")
            print(f"   Processing time: {best_easyocr['easyocr_time']:.2f}s")
            print(f"   Confidence: {best_easyocr['easyocr_confidence']:.1%}")
        else:
            print("\n🤝 Both OCR engines performed equally")
        
        print(f"\n💡 BEST ENHANCEMENT OVERALL: {best_overall['enhancement']}")
        print(f"   Contrast: {best_overall['contrast']:.1f}")
        
        return results
    
    def save_results(self, results, output_dir):
        """Save results to CSV and TXT files"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_path = output_dir / "ocr_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # Create detailed summary text file
        txt_path = output_dir / "summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FULL OCR COMPARISON RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for r in results:
                f.write(f"\n{'='*50}\n")
                f.write(f"ENHANCEMENT: {r['enhancement']}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Contrast (Std Dev): {r['contrast']:.2f}\n\n")
                
                f.write("TESSERACT RESULTS:\n")
                f.write(f"  Time: {r['tesseract_time']:.3f} seconds\n")
                f.write(f"  Characters: {r['tesseract_length']}\n")
                f.write(f"  Words: {r['tesseract_words']}\n")
                f.write(f"  Text: {r['tesseract_text'][:200]}\n\n")
                
                f.write("EASYOCR RESULTS:\n")
                f.write(f"  Time: {r['easyocr_time']:.3f} seconds\n")
                f.write(f"  Characters: {r['easyocr_length']}\n")
                f.write(f"  Words: {r['easyocr_words']}\n")
                f.write(f"  Confidence: {r['easyocr_confidence']:.2%}\n")
                f.write(f"  Detections: {r['easyocr_detections']}\n")
                f.write(f"  Text: {r['easyocr_text'][:200]}\n")
            
            # Add best results summary
            f.write(f"\n{'='*70}\n")
            f.write("BEST RESULTS SUMMARY\n")
            f.write(f"{'='*70}\n")
            
            best_tesseract = max(results, key=lambda x: x["tesseract_length"])
            best_easyocr = max(results, key=lambda x: x["easyocr_length"])
            
            f.write(f"\nBest for Tesseract: {best_tesseract['enhancement']}\n")
            f.write(f"  Text length: {best_tesseract['tesseract_length']} chars\n")
            f.write(f"  Time: {best_tesseract['tesseract_time']:.3f}s\n")
            
            f.write(f"\nBest for EasyOCR: {best_easyocr['enhancement']}\n")
            f.write(f"  Text length: {best_easyocr['easyocr_length']} chars\n")
            f.write(f"  Time: {best_easyocr['easyocr_time']:.3f}s\n")
            f.write(f"  Confidence: {best_easyocr['easyocr_confidence']:.2%}\n")
        
        print(f"💾 Summary saved to: {txt_path}")
    
    def run_full_pipeline(self, image_path, save_results=True):
        """Run complete pipeline"""
        
        print("\n" + "="*70)
        print("FULL OCR COMPARISON PIPELINE")
        print("="*70)
        print(f"Input image: {image_path}")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"❌ Error: Image not found at {image_path}")
            return None
        
        # Step 1: Create all enhancement versions
        versions, names = self.enhance_image_fully(image_path)
        
        # Step 2: Run OCR on all versions
        results = self.run_ocr_on_all_versions(versions, names)
        
        # Step 3: Compare and display
        self.compare_and_display(results)
        
        # Step 4: Save results to tests/full_ocr_comparison directory
        if save_results:
            # Save in tests/full_ocr_comparison directory
            output_dir = Path(__file__).parent / "full_ocr_comparison"
            self.save_results(results, output_dir)
        
        return results


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("FULL OCR COMPARISON TOOL")
    print("="*70)
    
    # Get image path from user
    image_path = input("\n📸 Enter path to your image: ").strip()
    image_path = image_path.strip('"').strip("'")
    
    if not image_path:
        # Use default test image
        default_image = Path(__file__).parent / "test_images" / "testocr.png"
        if default_image.exists():
            image_path = str(default_image)
        else:
            print("No default image found. Please provide an image path.")
            return
    
    print(f"Using image: {image_path}")
    
    # Run pipeline
    comparator = FullOCRComparison()
    results = comparator.run_full_pipeline(image_path, save_results=True)
    
    if results:
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Show where results are saved
        output_dir = Path(__file__).parent / "full_ocr_comparison"
        print(f"\n📁 Results saved in: {output_dir}")
        print(f"   - ocr_comparison_results.csv")
        print(f"   - summary.txt")


if __name__ == "__main__":
    main()