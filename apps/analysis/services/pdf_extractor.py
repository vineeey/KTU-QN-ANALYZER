"""
Comprehensive PDF extraction service using pdfplumber, PyMuPDF, and OCR.
Implements the exact specification from master prompt.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    PDF extraction service following master prompt specification:
    - Primary: pdfplumber
    - Secondary: PyMuPDF (fitz)
    - Fallback: OCR (pytesseract) for scanned PDFs only
    """
    
    def __init__(self):
        self.pdfplumber = None
        self.fitz = None
        self.pytesseract = None
        self.Image = None
        self._load_libraries()
    
    def _load_libraries(self):
        """Load PDF processing libraries with graceful fallback."""
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            logger.info("✓ pdfplumber loaded")
        except ImportError:
            logger.warning("pdfplumber not available")
        
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            logger.info("✓ PyMuPDF loaded")
        except ImportError:
            logger.warning("PyMuPDF not available")
        
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
            logger.info("✓ pytesseract loaded (OCR available)")
        except ImportError:
            logger.warning("pytesseract not available - OCR disabled")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract complete text from PDF using best available method.
        Priority: pdfplumber → PyMuPDF → OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        # Try pdfplumber first (most accurate for text-based PDFs)
        if self.pdfplumber:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text and len(text.strip()) > 100:  # Minimum viable text
                    logger.info(f"✓ Extracted {len(text)} chars using pdfplumber")
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Try PyMuPDF as backup
        if self.fitz:
            try:
                text = self._extract_with_pymupdf(pdf_path)
                if text and len(text.strip()) > 100:
                    logger.info(f"✓ Extracted {len(text)} chars using PyMuPDF")
                    return text
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")
        
        # Try OCR as last resort (for scanned PDFs)
        if self.pytesseract and self.Image:
            try:
                text = self._extract_with_ocr(pdf_path)
                if text:
                    logger.info(f"✓ Extracted {len(text)} chars using OCR")
                    return text
            except Exception as e:
                logger.error(f"OCR failed: {e}")
        
        raise RuntimeError("All extraction methods failed")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (primary method)."""
        text_parts = []
        
        with self.pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean up common PDF extraction artifacts
                    page_text = self._clean_extracted_text(page_text)
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean PDF extraction artifacts."""
        import re
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix broken words (e.g., "vulnerabil ity" -> "vulnerability")
        text = re.sub(r'(\w+)\s+(\w{1,3})\b', r'\1\2', text)
        
        # Remove standalone single characters that are artifacts (except 'a', 'I')
        text = re.sub(r'\b[J|j]\s+[J|j]\b', '', text)  # Remove "J J"
        text = re.sub(r'\s+[|]\s+', ' ', text)  # Remove pipe artifacts
        
        # Fix line breaks in middle of sentences
        text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (secondary method)."""
        text_parts = []
        
        doc = self.fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                page_text = self._clean_extracted_text(page_text)
                text_parts.append(page_text)
        doc.close()
        
        return "\n\n".join(text_parts)
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (fallback for scanned PDFs)."""
        text_parts = []
        
        # Convert PDF to images and run OCR
        doc = self.fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(matrix=self.fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Run OCR
            image = self.Image.open(io.BytesIO(img_data))
            page_text = self.pytesseract.image_to_string(image, lang='eng')
            
            if page_text:
                text_parts.append(page_text)
        
        doc.close()
        return "\n\n".join(text_parts)
    
    def extract_questions_with_metadata(self, pdf_path: str, year: int, session: str) -> List[Dict[str, Any]]:
        """
        Extract questions with metadata (year, session, marks, question number).
        
        Args:
            pdf_path: Path to PDF file
            year: Exam year
            session: Exam session (e.g., "December", "May")
            
        Returns:
            List of question dictionaries with metadata
        """
        text = self.extract_text(pdf_path)
        questions = self._parse_questions(text, year, session)
        return questions
    
    def _parse_questions(self, text: str, year: int, session: str) -> List[Dict[str, Any]]:
        """
        Parse questions from extracted text using regex patterns.
        Extracts question number, text, marks, and part (A/B).
        """
        questions = []
        
        # Pattern for KTU question papers
        # Matches: "Qn 1." or "1)" or "Q1." etc., followed by text and marks
        patterns = [
            # PART A pattern: Qn X. Question text (3 marks)
            r'(?:Qn|Q|Question)?\s*(\d+)[\.\)]\s*(.+?)(?:\((\d+)\s*marks?\))',
            # PART B pattern: Qn X(a) Question text (8 marks)
            r'(?:Qn|Q|Question)?\s*(\d+)[\(\[](a|b)[\)\]]\s*(.+?)(?:\((\d+)\s*marks?\))',
        ]
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 3:  # Simple question
                        qn, question_text, marks = match.groups()
                        part = 'A' if int(marks) <= 3 else 'B'
                        sub_part = None
                    else:  # Question with sub-parts
                        qn, sub_part, question_text, marks = match.groups()
                        part = 'B'  # Sub-parts are typically PART B
                    
                    questions.append({
                        'question_number': int(qn),
                        'sub_part': sub_part,
                        'text': question_text.strip(),
                        'marks': int(marks),
                        'part': part,
                        'year': year,
                        'session': session,
                        'full_identifier': f"Qn {qn}{f'({sub_part})' if sub_part else ''}"
                    })
                    break
        
        logger.info(f"Parsed {len(questions)} questions from PDF")
        return questions


class QuestionSegmenter:
    """
    Segments extracted text into individual questions using Python regex.
    Handles PART A and PART B separately.
    """
    
    def __init__(self):
        self.part_a_pattern = re.compile(
            r'(?:Qn|Q|Question)?\s*(\d+)[\.\)]\s*(.+?)(?:\((\d+)\s*marks?\))',
            re.IGNORECASE | re.DOTALL
        )
        self.part_b_pattern = re.compile(
            r'(?:Qn|Q|Question)?\s*(\d+)(?:[\(\[](a|b)[\)\]])?\s*(.+?)(?:\((\d+)\s*marks?\))',
            re.IGNORECASE | re.DOTALL
        )
    
    def segment(self, text: str) -> Dict[str, List[str]]:
        """
        Segment text into PART A and PART B questions.
        
        Returns:
            Dict with 'part_a' and 'part_b' lists
        """
        # Split by PART markers
        parts = re.split(r'PART\s*[AB]', text, flags=re.IGNORECASE)
        
        part_a_questions = []
        part_b_questions = []
        
        for i, section in enumerate(parts):
            if 'PART A' in text[max(0, text.find(section) - 20):text.find(section) + 10].upper():
                part_a_questions.extend(self._extract_questions(section, 'A'))
            elif 'PART B' in text[max(0, text.find(section) - 20):text.find(section) + 10].upper():
                part_b_questions.extend(self._extract_questions(section, 'B'))
        
        return {
            'part_a': part_a_questions,
            'part_b': part_b_questions
        }
    
    def _extract_questions(self, section: str, part: str) -> List[str]:
        """Extract individual questions from a section."""
        questions = []
        pattern = self.part_a_pattern if part == 'A' else self.part_b_pattern
        
        for match in pattern.finditer(section):
            question_text = match.group(0).strip()
            questions.append(question_text)
        
        return questions
