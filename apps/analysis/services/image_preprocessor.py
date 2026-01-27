"""
Advanced image preprocessing for better OCR quality.
Uses OpenCV for denoising, deskewing, and enhancement.
"""
import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Advanced image preprocessing to improve OCR accuracy.
    
    Features:
    - Denoising using fastNlMeansDenoisingColored
    - Adaptive thresholding for better text extraction
    - Deskewing to fix rotated scans
    - Morphological operations to clean up noise
    """
    
    def __init__(self):
        self.cv2 = None
        self._load_opencv()
    
    def _load_opencv(self):
        """Lazy load OpenCV."""
        try:
            import cv2
            self.cv2 = cv2
            logger.info("✓ OpenCV loaded for image preprocessing")
        except ImportError:
            logger.warning("OpenCV not available - advanced preprocessing disabled")
    
    def preprocess(
        self,
        image: Image.Image,
        denoise: bool = True,
        deskew: bool = True,
        adaptive_threshold: bool = True,
        morph_clean: bool = True
    ) -> Image.Image:
        """
        Apply full preprocessing pipeline to image.
        
        Args:
            image: PIL Image object
            denoise: Apply denoising
            deskew: Fix rotation
            adaptive_threshold: Apply adaptive thresholding
            morph_clean: Apply morphological cleaning
            
        Returns:
            Preprocessed PIL Image
        """
        if not self.cv2:
            return image
        
        # Convert PIL to OpenCV format
        img_cv = self._pil_to_cv(image)
        
        # Apply preprocessing steps
        if denoise:
            img_cv = self._denoise(img_cv)
        
        if deskew:
            img_cv = self._deskew(img_cv)
        
        if adaptive_threshold:
            img_cv = self._adaptive_threshold(img_cv)
        
        if morph_clean:
            img_cv = self._morphological_clean(img_cv)
        
        # Convert back to PIL
        return self._cv_to_pil(img_cv)
    
    def _pil_to_cv(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        return self.cv2.cvtColor(img_array, self.cv2.COLOR_RGB2BGR)
    
    def _cv_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        # Convert BGR to RGB
        if len(cv_image.shape) == 3:
            rgb_image = self.cv2.cvtColor(cv_image, self.cv2.COLOR_BGR2RGB)
        else:
            # Grayscale
            rgb_image = cv_image
        
        return Image.fromarray(rgb_image)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce scan artifacts and noise.
        
        Uses fastNlMeansDenoisingColored for color images.
        """
        try:
            if len(image.shape) == 3:
                # Color image
                denoised = self.cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=10,
                    hColor=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:
                # Grayscale
                denoised = self.cv2.fastNlMeansDenoising(
                    image,
                    None,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            logger.debug("Applied denoising")
            return denoised
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image rotation (deskew).
        
        Uses Hough transform to detect text lines and calculate skew angle.
        """
        try:
            # Convert to grayscale
            gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply edge detection
            edges = self.cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = self.cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if -45 < angle < 45:  # Only consider reasonable angles
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only deskew if angle is significant (> 0.5 degrees)
                    if abs(median_angle) > 0.5:
                        # Get image dimensions
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        
                        # Create rotation matrix
                        M = self.cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        
                        # Rotate image
                        rotated = self.cv2.warpAffine(
                            image,
                            M,
                            (w, h),
                            flags=self.cv2.INTER_CUBIC,
                            borderMode=self.cv2.BORDER_REPLICATE
                        )
                        
                        logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
                        return rotated
            
            return image
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for better text extraction.
        
        Better than simple binary threshold as it adapts to local lighting conditions.
        """
        try:
            # Convert to grayscale
            gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply adaptive thresholding
            thresh = self.cv2.adaptiveThreshold(
                gray,
                255,
                self.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                self.cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            
            # Convert back to BGR for consistency
            result = self.cv2.cvtColor(thresh, self.cv2.COLOR_GRAY2BGR)
            
            logger.debug("Applied adaptive thresholding")
            return result
        except Exception as e:
            logger.warning(f"Adaptive thresholding failed: {e}")
            return image
    
    def _morphological_clean(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up noise.
        
        Uses opening (erosion followed by dilation) to remove small noise.
        """
        try:
            # Convert to grayscale
            gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Create morphological kernel
            kernel = self.cv2.getStructuringElement(self.cv2.MORPH_RECT, (2, 2))
            
            # Apply morphological opening
            cleaned = self.cv2.morphologyEx(gray, self.cv2.MORPH_OPEN, kernel)
            
            # Convert back to BGR
            result = self.cv2.cvtColor(cleaned, self.cv2.COLOR_GRAY2BGR)
            
            logger.debug("Applied morphological cleaning")
            return result
        except Exception as e:
            logger.warning(f"Morphological cleaning failed: {e}")
            return image
    
    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Convenience method: Apply optimal preprocessing for OCR.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image optimized for OCR
        """
        return self.preprocess(
            image,
            denoise=True,
            deskew=True,
            adaptive_threshold=True,
            morph_clean=True
        )
    
    def quick_enhance(self, image: Image.Image) -> Image.Image:
        """
        Quick preprocessing with minimal operations.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image with basic enhancements
        """
        return self.preprocess(
            image,
            denoise=True,
            deskew=False,
            adaptive_threshold=True,
            morph_clean=False
        )
