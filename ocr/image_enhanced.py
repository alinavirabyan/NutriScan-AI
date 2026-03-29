import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path  # <-- IMPORTANT: Added this import
import logging

logger = logging.getLogger(__name__)

class ImageEnhancer:
    """Class for image enhancement"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def to_grayscale(self, img: Image.Image) -> Image.Image:
        """Convert to grayscale"""
        return img.convert("L")
    
    def remove_noise(self, img: Image.Image, strength: int = 3) -> Image.Image:
        """Remove noise"""
        arr = np.array(img)
        if strength % 2 == 0:
            strength += 1
        blurred = cv2.GaussianBlur(arr, (strength, strength), 0)
        return Image.fromarray(blurred)
    
    def auto_contrast(self, img: Image.Image) -> Image.Image:
        """Auto contrast correction"""
        return ImageOps.autocontrast(img)
    
    def otsu_threshold(self, img: Image.Image) -> Image.Image:
        """Otsu threshold binarization"""
        arr = np.array(img)
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)
    
    def deskew(self, img: Image.Image) -> Image.Image:
        """Straighten tilted text"""
        arr = np.array(img)
        coords = np.column_stack(np.where(arr < 128))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return img
        (h, w) = arr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC)
        return Image.fromarray(rotated)
    
    def sharpen(self, img: Image.Image, factor: float = 1.5) -> Image.Image:
        """Sharpen image"""
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def upscale(self, img: Image.Image, min_height: int = 64) -> Image.Image:
        """Upscale small images"""
        if img.height < min_height:
            scale = min_height / img.height
            new_w = int(img.width * scale)
            img = img.resize((new_w, min_height), Image.LANCZOS)
        return img
    
    def enhance_image(self, image_input, **kwargs) -> tuple:
        """
        Main image enhancement method
        
        Args:
            image_input: PIL Image or file path
            
        Returns:
            tuple: (enhanced_image, saved_path)
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
            original_path = str(image_input)
        else:
            img = image_input.copy()
            original_path = None
        
        # Apply enhancements
        img = self.to_grayscale(img)
        img = self.upscale(img, min_height=64)
        img = self.remove_noise(img, strength=3)
        img = self.auto_contrast(img)
        img = self.sharpen(img, factor=1.5)
        
        try:
            img = self.deskew(img)
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
        
        img = self.otsu_threshold(img)
        
        # Save enhanced image
        if original_path:
            p = Path(original_path)
            enhanced_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")
            img.save(enhanced_path)
            logger.info(f"Enhanced image saved: {enhanced_path}")
            return img, enhanced_path
        
        return img, None


# For backward compatibility
def preprocess_image(*args, **kwargs):
    """Wrapper for backward compatibility"""
    enhancer = ImageEnhancer()
    img, _ = enhancer.enhance_image(*args, **kwargs)
    return img