"""
PaddleOCR-based OCR engine for scanned PDF pages.

Provides text extraction with confidence scoring, per-page retry logic,
and optional debug output storage.
"""
import logging
import io
import re
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRResult:
    """Value object holding OCR output for a single page."""

    def __init__(self, page_number: int, text: str, confidence: float, raw_lines: list):
        self.page_number = page_number
        self.text = text
        self.confidence = confidence         # Mean confidence across detected lines
        self.raw_lines = raw_lines           # Raw PaddleOCR line output

    def __repr__(self) -> str:
        return (
            f"OCRResult(page={self.page_number}, "
            f"confidence={self.confidence:.2f}, chars={len(self.text)})"
        )


class OCREngine:
    """
    PaddleOCR wrapper with confidence-aware retry and optional debug output.

    Usage::

        engine = OCREngine()
        results = engine.process_pdf(pdf_path)
        for r in results:
            print(r.text)

    Design decisions:
    - Singleton PaddleOCR instance (lazy-loaded) to avoid per-request overhead.
    - Low-confidence pages are re-processed with enhanced preprocessing.
    - Debug images are optionally saved to a configurable directory.
    """

    # Module-level singleton – shared across all instances in the process.
    _paddle_ocr = None
    _ocr_loaded: bool = False

    # Confidence below which a page will be retried with extra preprocessing.
    LOW_CONFIDENCE_THRESHOLD: float = 0.60

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        debug_dir: Optional[str] = None,
    ):
        """
        Args:
            lang: PaddleOCR language code (default 'en').
            use_angle_cls: Enable angle classification for rotated text.
            debug_dir: If set, save intermediate images to this directory.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._ensure_ocr_loaded()

    # ------------------------------------------------------------------
    # Lazy singleton loader
    # ------------------------------------------------------------------

    def _ensure_ocr_loaded(self) -> None:
        """
        Initialise the PaddleOCR backend once per process.

        PaddleOCR is the primary and only OCR engine. It provides
        high-quality text recognition with built-in angle classification
        and multi-language support.
        """
        if OCREngine._ocr_loaded:
            return

        try:
            from paddleocr import PaddleOCR
            OCREngine._paddle_ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                show_log=False,
            )
            OCREngine._ocr_loaded = True
            logger.info("✓ OCR backend: PaddleOCR (lang=%s, angle_cls=%s)",
                        self.lang, self.use_angle_cls)
        except ImportError:
            logger.warning(
                "PaddleOCR not available. "
                "Install with: pip install paddleocr paddlepaddle"
            )
        except Exception as exc:
            logger.error("Failed to initialise PaddleOCR: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> List[OCRResult]:
        """
        Extract text from a PDF using the best available strategy.

        Strategy (ordered by quality):
        1. If the PDF has a clean text layer (digital PDF), extract directly.
        2. If the text layer exists but has OCR artifacts (previously badly
           OCR-d scans), render to images and re-OCR with PaddleOCR.
        3. Purely image-based PDFs (no text layer) also go through PaddleOCR.

        After extraction every page's text is also run through the domain-aware
        artifact cleaner before being returned.
        """
        quality = self._assess_text_quality(pdf_path)

        if quality == "clean":
            logger.info("Clean text PDF – extracting directly: %s", pdf_path)
            results = self._extract_text_directly(pdf_path)
        else:
            if quality == "dirty":
                logger.info(
                    "Dirty/garbled text layer detected – forcing image OCR: %s", pdf_path
                )
            else:
                logger.info("Image-only PDF – using PaddleOCR: %s", pdf_path)
            images = self._pdf_to_images(pdf_path)
            results = []
            for page_num, img in enumerate(images, start=1):
                result = self._ocr_image(img, page_num)
                if result.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                    result = self._retry_with_preprocessing(img, page_num)
                results.append(result)
                if self.debug_dir:
                    self._save_debug_image(img, page_num)

        # Post-process: apply domain-specific artifact cleaner to every page
        for r in results:
            r.text = self._clean_page_text(r.text)

        logger.info("Extraction complete: %d pages from %s", len(results), pdf_path)
        return results

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    # Patterns that indicate a bad/previously-OCR'd text layer
    _ARTIFACT_PATTERNS = [
        re.compile(r"(?<!\d)\b3\b(?!\d)"),        # stray lone digit 3
        re.compile(r"\bJ\s"),                      # lone J followed by space
        re.compile(r"\b[a-z]{15,}\b"),             # fused words (no spaces)
        re.compile(r"\b[A-Za-z]{2,4}-[a-z]{2,4},"),# hyphen OCR artefact e.g. ltaz-afi,
        re.compile(r"[a-z]\.[a-z]"),               # missing space after period
    ]

    def _assess_text_quality(self, pdf_path: str) -> str:
        """
        Return one of:
        - ``"clean"``   – PDF has a reliable digital text layer
        - ``"dirty"``   – PDF has a text layer but it's garbled/noisy
        - ``"image"``   – PDF has no usable text layer (pure image)

        Heuristic:
        1. Extract text; if average chars/page < 50 → "image"
        2. Score artifact density (stray digits, lone J, fused words, etc.)
           If artifact_ratio > 0.5 per-100-chars → "dirty"
        3. Otherwise → "clean"
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            pages_text = [p.get_text("text") for p in doc]
            doc.close()
        except Exception:
            return "image"

        total_chars = sum(len(t) for t in pages_text)
        n_pages = max(len(pages_text), 1)
        avg_chars = total_chars / n_pages

        if avg_chars < 50:
            return "image"

        # Count artifacts across whole document
        all_text = "\n".join(pages_text)
        artifact_hits = sum(
            len(pat.findall(all_text)) for pat in self._ARTIFACT_PATTERNS
        )
        # Normalise: artifacts per 100 characters
        artifact_ratio = (artifact_hits * 100) / max(total_chars, 1)

        if artifact_ratio > 0.4:
            logger.debug(
                "Artifact ratio %.2f > 0.4 for %s — routing to image OCR",
                artifact_ratio, pdf_path
            )
            return "dirty"

        return "clean"

    # ------------------------------------------------------------------
    # Domain-aware page-level artifact cleaner
    # ------------------------------------------------------------------

    # Common OCR misreads in KTU/disaster-management exam papers
    _OCR_WORD_FIXES = [
        # Fused / mangled words
        (re.compile(r"\bhazardmapping\b", re.I),    "hazard mapping"),
        (re.compile(r"\bltaz[-]?afi\b", re.I),      "hazard"),
        (re.compile(r"\bIdentiff\b"),                "Identify"),
        (re.compile(r"\bdentifu\b", re.I),           "Identify"),
        (re.compile(r"\bdentiff\b", re.I),           "Identify"),
        (re.compile(r"\bIdentify\b"),                "Identify"),
        (re.compile(r"\brespecf\b", re.I),           "respect"),
        (re.compile(r"\brecovery-and\b", re.I),      "recovery and"),
        (re.compile(r"\bth\.e\b", re.I),             "the"),
        (re.compile(r"\bthev\b"),                    "they"),
        (re.compile(r"\bcan g \b", re.I),            "can "),
        (re.compile(r"\bStatenthe\b", re.I),         "State the"),
        (re.compile(r"\bStatethe\b", re.I),          "State the"),
        (re.compile(r"\blnstitute\b"),               "Institute"),
        (re.compile(r"\[he\b"),                      "the"),
        # Lone characters that are OCR noise
        (re.compile(r"\s+J\s+"),                     " "),    # lone J spacer
        (re.compile(r"\s+J$", re.M),                 ""),     # trailing J
        (re.compile(r"^J\s+", re.M),                 ""),     # leading J
        # Stray 3 in sentence context (not valid marks/numbers)
        (re.compile(r"(\b[a-zA-Z]+\b)\s+3\s+(\b[a-zA-Z]+\b)"),
                                                     r"\1 \2"),
        (re.compile(r"(\b[a-zA-Z]+\b)\s+3\s*$", re.M),
                                                     r"\1"),
        # Parentheses artifacts:  "a )" → remove
        (re.compile(r"\s+[a-c]\s*\)(?!\s*[A-Za-z])"), ""),
        # Misread opening letters (capital I misread as lowercase n/l)
        (re.compile(r"^• n "),                       "• In "),
        (re.compile(r"^• l "),                       "• I "),
    ]

    def _clean_page_text(self, text: str) -> str:
        """Apply domain-specific word-level fixes to a page of text."""
        for pattern, replacement in self._OCR_WORD_FIXES:
            text = pattern.sub(replacement, text)
        # Collapse multiple spaces created by removals
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove lines that are now empty or only punctuation
        lines = [ln for ln in text.splitlines() if re.search(r"[A-Za-z0-9]", ln)]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Direct text extraction (clean PDFs only)
    # ------------------------------------------------------------------

    def _extract_text_directly(self, pdf_path: str) -> List[OCRResult]:
        """
        Extract text verbatim from each page of a digital PDF.

        Returns OCRResult objects with confidence=1.0 (no OCR uncertainty).
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            results: List[OCRResult] = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")  # plain text, preserves newlines
                lines = [ln for ln in text.split("\n") if ln.strip()]
                results.append(
                    OCRResult(
                        page_number=page_num,
                        text=text,
                        confidence=1.0,
                        raw_lines=lines,
                    )
                )
            doc.close()
            logger.info(
                "Direct extraction complete: %d pages from %s", len(results), pdf_path
            )
            return results
        except Exception as exc:
            logger.error("Direct text extraction failed: %s – falling back to OCR", exc)
            # Fall back to the image-based pipeline
            images = self._pdf_to_images(pdf_path)
            return [self._ocr_image(img, i + 1) for i, img in enumerate(images)]

    def process_image(self, image, page_number: int = 1) -> OCRResult:
        """
        Run OCR on a single PIL Image object.

        Args:
            image: PIL.Image instance.
            page_number: Label used in the returned OCRResult.

        Returns:
            OCRResult for the image.
        """
        result = self._ocr_image(image, page_number)
        if result.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            result = self._retry_with_preprocessing(image, page_number)
        return result

    # ------------------------------------------------------------------
    # PDF → images
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: str):
        """
        Render each PDF page to a PIL Image at 300 DPI using PyMuPDF.

        Returns:
            List of PIL.Image objects.
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image

            doc = fitz.open(pdf_path)
            images = []
            mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img.convert("RGB"))
            doc.close()
            logger.debug("Rendered %d pages from %s at 300 DPI", len(images), pdf_path)
            return images
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed – cannot render PDF pages")
            return []
        except Exception as exc:
            logger.error("PDF rendering failed for %s: %s", pdf_path, exc)
            return []

    # ------------------------------------------------------------------
    # Core OCR
    # ------------------------------------------------------------------

    def _ocr_image(self, image, page_number: int) -> OCRResult:
        """Run PaddleOCR on a PIL Image."""
        if OCREngine._paddle_ocr:
            return self._ocr_image_paddle(image, page_number)
        logger.warning("PaddleOCR backend not available for page %d", page_number)
        return OCRResult(page_number, "", 0.0, [])

    def _ocr_image_paddle(self, image, page_number: int) -> OCRResult:
        """Run PaddleOCR on a PIL Image and return structured OCRResult."""
        try:
            import numpy as np
            img_array = np.array(image)
            raw = OCREngine._paddle_ocr.ocr(img_array)

            if not raw or raw[0] is None:
                return OCRResult(page_number, "", 0.0, [])

            lines = raw[0]
            texts: List[str] = []
            confidences: List[float] = []

            for line in lines:
                if not line or len(line) < 2:
                    continue
                text_conf = line[1]
                if text_conf and len(text_conf) == 2:
                    texts.append(text_conf[0])
                    confidences.append(float(text_conf[1]))

            full_text = "\n".join(texts)
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return OCRResult(
                page_number=page_number,
                text=full_text,
                confidence=mean_confidence,
                raw_lines=lines,
            )
        except Exception as exc:
            logger.error("PaddleOCR failed on page %d: %s", page_number, exc)
            return OCRResult(page_number, "", 0.0, [])

    def _retry_with_preprocessing(self, image, page_number: int) -> OCRResult:
        """
        Re-run OCR after applying enhanced image preprocessing.

        Used when initial OCR confidence is below LOW_CONFIDENCE_THRESHOLD.
        """
        try:
            from .image_preprocessor import (
                to_grayscale,
                apply_median_blur,
                apply_sharpen_filter,
                adaptive_gaussian_threshold,
                morphological_closing,
            )

            enhanced = to_grayscale(image)
            enhanced = apply_median_blur(enhanced)
            enhanced = apply_sharpen_filter(enhanced)
            enhanced = adaptive_gaussian_threshold(enhanced)
            enhanced = morphological_closing(enhanced)

            return self._ocr_image(enhanced, page_number)
        except Exception as exc:
            logger.warning("Preprocessing retry failed on page %d: %s", page_number, exc)
            return self._ocr_image(image, page_number)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _save_debug_image(self, image, page_number: int) -> None:
        """Save a page image to debug_dir for inspection."""
        try:
            if self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                path = self.debug_dir / f"page_{page_number:03d}.png"
                image.save(str(path))
        except Exception as exc:
            logger.debug("Could not save debug image for page %d: %s", page_number, exc)
