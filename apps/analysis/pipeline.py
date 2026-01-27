"""
Enhanced analysis pipeline with dual classification system.
"""
import logging
import os
from typing import Optional
from django.utils import timezone
from django.conf import settings

from apps.papers.models import Paper
from apps.questions.models import Question
from apps.analytics.clustering import TopicClusteringService
from .models import AnalysisJob
from .services.pymupdf_extractor import PyMuPDFExtractor
from .services.extractor import QuestionExtractor
from .services.classifier import ModuleClassifier
from .services.bloom import BloomClassifier
from .services.difficulty import DifficultyEstimator
from .services.hybrid_llm_service import HybridLLMService
from .services.image_preprocessor import ImagePreprocessor
import fitz  # PyMuPDF for rendering (already a dependency)

# Services with optional numpy dependency
try:
    from .services.ai_classifier import AIClassifier
    from .services.embedder import EmbeddingService
    from .services.similarity import SimilarityService
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    AIClassifier = None
    EmbeddingService = None
    SimilarityService = None

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Enhanced orchestration of analysis workflow with dual classification.
    - KTU: Rule-based mapping (strict)
    - Other: AI-based classification (LLM + embeddings + clustering)
    """
    
    def __init__(self, llm_client=None):
        # Extractors
        self.pymupdf_extractor = PyMuPDFExtractor()  # Primary extractor
        self.fallback_extractor = QuestionExtractor()  # Fallback
        
        # Hybrid LLM Service for OCR cleaning and similarity
        self.hybrid_llm = HybridLLMService()
        self.image_preprocessor = ImagePreprocessor()
        
        # Services (with numpy fallback handling)
        if NUMPY_AVAILABLE:
            self.embedder = EmbeddingService()
            self.similarity = SimilarityService(threshold=0.80)
            self.ai_classifier = AIClassifier(llm_client, self.embedder)
        else:
            self.embedder = None
            self.similarity = None
            self.ai_classifier = None
            logger.warning("NumPy not available - AI features disabled")
        
        self.bloom_classifier = BloomClassifier(llm_client)
        self.difficulty_estimator = DifficultyEstimator(llm_client)
        
        # Classifiers
        self.module_classifier = ModuleClassifier(llm_client)  # For KTU
        
        self.llm_client = llm_client
    
    def analyze_paper(self, paper: Paper) -> AnalysisJob:
        """
        Run complete analysis on a paper with dual classification support.
        
        Args:
            paper: Paper instance to analyze
            
        Returns:
            AnalysisJob with results
        """
        # Create analysis job
        job = AnalysisJob.objects.create(paper=paper)
        job.started_at = timezone.now()
        job.save()
        
        try:
            subject = paper.subject
            is_ktu = subject.university_type == 'KTU' if hasattr(subject, 'university_type') else True
            
            logger.info(f"Starting analysis for {paper.title} - University: {subject.university_type if hasattr(subject, 'university_type') else 'KTU'}")
            
            # Step 1: Extract text and questions (pdfplumber first, then fitz, then OCR)
            job.status = AnalysisJob.Status.EXTRACTING
            job.progress = 5
            job.status_detail = 'Reading PDF file...'
            job.save()
            
            paper.status = Paper.ProcessingStatus.PROCESSING
            paper.status_detail = 'Extracting text from PDF (pdfplumber)...'
            paper.progress_percent = 5
            paper.save()

            questions_data = []
            images = []

            # Primary: pdfplumber via QuestionExtractor
            primary_text = self.fallback_extractor.extract_text(paper.file.path)
            if primary_text:
                questions_data = self.fallback_extractor.extract_questions(primary_text)
                paper.raw_text = primary_text
                paper.status_detail = f'Extracted {len(questions_data)} questions (pdfplumber)'
                paper.questions_extracted = len(questions_data)
                paper.progress_percent = 15
                paper.save()

            # Fallback: PyMuPDF structured parse (with images)
            if not questions_data:
                paper.status_detail = 'Extracting text from PDF (PyMuPDF fallback)...'
                paper.save()
                try:
                    questions_data, images = self.pymupdf_extractor.extract_questions_with_images(
                        paper.file.path
                    )
                    paper.raw_text = self.pymupdf_extractor.extract_text(paper.file.path)
                    paper.page_count = self.pymupdf_extractor.get_page_count(paper.file.path)
                    paper.status_detail = f'Extracted {len(questions_data)} questions (fitz)'
                    paper.questions_extracted = len(questions_data)
                    paper.progress_percent = 20
                    paper.save()
                    logger.info(f"PyMuPDF: Extracted {len(questions_data)} questions and {len(images)} images")
                except Exception as e:
                    logger.error(f"PyMuPDF extraction failed: {e}")
                    images = []

            # OCR fallback if still empty
            if not questions_data:
                logger.warning("No questions extracted; attempting OCR fallback")
                ocr_questions, ocr_text = self._ocr_extract_questions(paper.file.path)
                if ocr_questions:
                    questions_data = ocr_questions
                    paper.raw_text = ocr_text or paper.raw_text
                    paper.status_detail = f'Extracted {len(questions_data)} questions (OCR)'
                    paper.questions_extracted = len(questions_data)
                    paper.save()

            if not questions_data:
                raise Exception("No questions could be extracted from this PDF (try a clearer PDF or OCR)")
            
            job.questions_extracted = len(questions_data)
            job.progress = 30
            job.status_detail = f'Found {len(questions_data)} questions from {paper.page_count} pages'
            job.save()
            
            # Step 2: Classify questions based on university type
            job.status = AnalysisJob.Status.CLASSIFYING
            job.progress = 40
            job.status_detail = 'Creating question records...'
            job.save()
            
            modules = list(subject.modules.all())
            
            if is_ktu:
                # KTU: Use strict rule-based classification
                classified_questions = self._classify_ktu_questions(
                    questions_data, subject, modules
                )
            else:
                # Other Universities: Use AI-based classification
                if self.ai_classifier:
                    syllabus_text = subject.syllabus_text if hasattr(subject, 'syllabus_text') else None
                    classified_questions = self.ai_classifier.classify_questions_semantic(
                        questions_data, subject, syllabus_text
                    )
                else:
                    # Fallback to KTU classification if AI not available
                    logger.warning("AI classifier not available, using KTU classification")
                    classified_questions = self._classify_ktu_questions(
                        questions_data, subject, modules
                    )
            
            job.progress = 60
            job.status_detail = f'Classified {len(classified_questions)} questions'
            job.save()

            # Surface classification progress on the paper for UI polling
            paper.questions_classified = len(classified_questions)
            paper.progress_percent = max(paper.progress_percent, 60)
            paper.status_detail = f'Classified {len(classified_questions)} questions'
            paper.save()
            
            # Step 3: Create question objects in database
            job.status = AnalysisJob.Status.ANALYZING
            job.progress = 65
            job.status_detail = 'Creating question records...'
            job.save()

            paper.progress_percent = max(paper.progress_percent, 65)
            paper.status_detail = 'Saving question records...'
            paper.save()
            
            created_questions = []
            for i, q_data in enumerate(classified_questions):
                # Update progress every 5 questions
                if i > 0 and i % 5 == 0:
                    progress = 60 + int((i / len(classified_questions)) * 20)
                    job.progress = progress
                    job.status_detail = f'Saving question {i}/{len(classified_questions)}...'
                    job.save()
                
                # Find module
                module = None
                if 'module_number' in q_data:
                    module = next(
                        (m for m in modules if m.number == q_data['module_number']), 
                        None
                    )
                
                # Create question
                question = Question.objects.create(
                    paper=paper,
                    question_number=q_data.get('question_number', ''),
                    text=q_data['text'],
                    marks=q_data.get('marks'),
                    part=q_data.get('part', ''),
                    module=module,
                    images=q_data.get('images', []),
                    question_type=q_data.get('question_type', ''),
                    difficulty=q_data.get('difficulty', ''),
                    bloom_level=q_data.get('bloom_level', ''),
                    embedding=q_data.get('embedding')
                )
                
                created_questions.append(question)

            # Step 4: Build/update topic clusters for the whole subject
            job.progress = 85
            job.status_detail = 'Building topic clusters...'
            job.save()

            paper.progress_percent = max(paper.progress_percent, 85)
            paper.status_detail = 'Building topic clusters...'
            paper.save()

            try:
                TopicClusteringService(subject).analyze_subject()
            except Exception as e:
                # Clustering is best-effort; do not fail paper analysis if clustering fails.
                logger.error(f"Topic clustering failed for subject {subject}: {e}", exc_info=True)
            
            job.progress = 90
            job.status_detail = 'Finalizing analysis...'
            job.save()
            
            # Step 4: Mark paper as completed
            paper.status = Paper.ProcessingStatus.COMPLETED
            paper.processed_at = timezone.now()
            paper.save()
            
            # Complete job
            job.status = AnalysisJob.Status.COMPLETED
            job.progress = 100
            job.status_detail = 'Analysis completed successfully'
            job.completed_at = timezone.now()
            job.save()

            paper.progress_percent = 100
            paper.status_detail = 'Analysis completed successfully'
            paper.save()
            
            logger.info(f"Analysis completed: {len(created_questions)} questions created")
            return job
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            
            # Mark as failed
            job.status = AnalysisJob.Status.FAILED
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
            
            paper.status = Paper.ProcessingStatus.FAILED
            paper.processing_error = str(e)
            paper.save()
            
            raise
    
    def _classify_ktu_questions(
        self,
        questions_data: list,
        subject,
        modules: list
    ) -> list:
        """
        KTU-specific rule-based classification.
        STRICT map:
            Part A: Q1–Q10 → (1,1,2,2,3,3,4,4,5,5)
            Part B: Q11–Q20 → (1,1,2,2,3,3,4,4,5,5)
        No AI inference.
        """
        logger.info("Using KTU strict rule-based classification")

        # Fixed maps for module assignment
        part_a_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        part_b_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

        classified = []

        for idx, q_data in enumerate(questions_data):
            q_number = idx + 1  # 1-indexed

            if q_number <= 10:
                part = 'A'
                module_num = part_a_modules[q_number - 1]
            else:
                part = 'B'
                offset = q_number - 11
                module_num = part_b_modules[offset] if 0 <= offset < len(part_b_modules) else part_b_modules[-1]

            q_data['question_number'] = str(q_number)
            q_data['part'] = part
            q_data['module_number'] = module_num

            # Rule-based Bloom/difficulty and simple type
            q_data['bloom_level'] = self.bloom_classifier.classify(q_data['text'])
            q_data['difficulty'] = self.difficulty_estimator.estimate(
                q_data['text'], q_data.get('marks')
            )
            q_data['question_type'] = self._simple_question_type(q_data['text'])

            classified.append(q_data)

        return classified
    
    def _simple_question_type(self, text: str) -> str:
        """Simple rule-based question type classification."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['define', 'what is']):
            return 'definition'
        elif any(word in text_lower for word in ['derive', 'proof']):
            return 'derivation'
        elif any(word in text_lower for word in ['calculate', 'compute']):
            return 'numerical'
        elif any(word in text_lower for word in ['draw', 'diagram']):
            return 'diagram'
        elif any(word in text_lower for word in ['compare', 'differentiate']):
            return 'comparison'
        else:
            return 'theory'

    def _ocr_extract_questions(self, pdf_path: str):
        """Render pages to images and OCR text, then parse questions.

        Returns: (questions, raw_text)
        """
        try:
            import pytesseract
            from PIL import Image, ImageOps, ImageEnhance, ImageFilter
            # Explicitly set tesseract path on Windows when not on PATH
            tess_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            if os.path.exists(tess_cmd):
                pytesseract.pytesseract.tesseract_cmd = tess_cmd
        except Exception as e:
            logger.warning(f"OCR skipped: pytesseract/Pillow not available ({e})")
            return [], None

        ocr_text_parts = []
        raw_ocr_pages = []  # Store raw OCR for cleaning

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"OCR failed to open PDF: {e}")
            return [], None

        try:
            for page_idx, page in enumerate(doc):
                # Render page to image at higher DPI
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Use advanced preprocessing if enabled
                if settings.OCR_ENHANCEMENT.get('USE_ADVANCED_PREPROCESSING', False):
                    try:
                        img = self.image_preprocessor.enhance_for_ocr(img)
                        logger.debug(f"Applied advanced preprocessing to page {page_idx + 1}")
                    except Exception as e:
                        logger.warning(f"Image preprocessing failed: {e}")

                candidates = []

                # Base grayscale
                gray = ImageOps.grayscale(img)
                candidates.append(gray)

                # Contrast enhanced
                enhancer = ImageEnhance.Contrast(gray)
                candidates.append(enhancer.enhance(1.8))

                # Sharpened
                candidates.append(gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)))

                # Two thresholds
                candidates.append(gray.point(lambda x: 0 if x < 170 else 255, '1'))
                candidates.append(gray.point(lambda x: 0 if x < 200 else 255, '1'))

                page_text = []
                for cand in candidates:
                    try:
                        text = pytesseract.image_to_string(cand, config='--psm 6 --oem 3')
                    except Exception as e:
                        logger.warning(f"OCR engine error: {e}")
                        text = ''
                    if text and text.strip():
                        page_text.append(text)

                if page_text:
                    # Use the best candidate (longest text usually means most accurate)
                    best_text = max(page_text, key=len)
                    raw_ocr_pages.append(best_text)
                    ocr_text_parts.append(best_text)
        finally:
            doc.close()

        if not ocr_text_parts:
            logger.warning("OCR produced no text")
            return [], None

        # Apply LLM cleaning to OCR text if enabled
        if settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', True) and raw_ocr_pages:
            try:
                logger.info("Applying LLM-based OCR cleaning...")
                cleaned_pages, llm_used = self.hybrid_llm.clean_ocr_batch(
                    pages=raw_ocr_pages,
                    subject_name=None,  # TODO: Pass subject name if available
                    year=None  # TODO: Extract year from paper if available
                )
                
                if cleaned_pages and llm_used != 'none':
                    ocr_text_parts = cleaned_pages
                    logger.info(f"✓ OCR text cleaned using {llm_used}")
                    # Store raw OCR for debugging (will be used when creating questions)
            except Exception as e:
                logger.error(f"LLM OCR cleaning failed: {e}")
                # Continue with raw OCR if cleaning fails

        full_text = "\n".join(ocr_text_parts)
        questions = self.fallback_extractor.extract_questions(full_text)
        logger.info(f"OCR extracted {len(questions)} questions")
        return questions, full_text
