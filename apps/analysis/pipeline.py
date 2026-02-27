"""
Enhanced analysis pipeline with dual classification system.
"""
import logging
import os
import re
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
from .services.text_cleaner import TextCleaner as PipelineTextCleaner
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
        
        # Text cleaner for OCR artifact removal
        self.text_cleaner = PipelineTextCleaner()
        
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
                # Always apply deterministic text cleaning first
                primary_text = self.text_cleaner.clean(primary_text)
                logger.info("Applied deterministic text cleaning (text_cleaner)")
                
                # Clean extracted text using LLM if enabled
                if settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', True):
                    try:
                        paper.status_detail = 'Cleaning extracted text with LLM...'
                        paper.save()
                        logger.info("Applying LLM-based text cleaning...")
                        subject_name = paper.subject.name if hasattr(paper.subject, 'name') else None
                        year = str(paper.year) if hasattr(paper, 'year') else None
                        cleaned_text, llm_used = self.hybrid_llm.clean_ocr_text(
                            raw_text=primary_text,
                            subject_name=subject_name,
                            year=year,
                            use_advanced=True
                        )
                        if cleaned_text and llm_used != 'none':
                            primary_text = cleaned_text
                            logger.info(f"✓ Text cleaned using {llm_used}")
                        else:
                            logger.warning("LLM cleaning returned no result, using original text")
                    except Exception as e:
                        logger.error(f"LLM text cleaning failed: {e}, using original text")
                
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
                    raw_text = self.pymupdf_extractor.extract_text(paper.file.path)
                    
                    # Always apply deterministic text cleaning first
                    raw_text = self.text_cleaner.clean(raw_text)
                    logger.info("Applied deterministic text cleaning to PyMuPDF text")
                    
                    # Clean extracted text using LLM if enabled
                    if settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', True):
                        try:
                            paper.status_detail = 'Cleaning extracted text with LLM...'
                            paper.save()
                            logger.info("Applying LLM-based text cleaning...")
                            subject_name = paper.subject.name if hasattr(paper.subject, 'name') else None
                            year = str(paper.year) if hasattr(paper, 'year') else None
                            cleaned_text, llm_used = self.hybrid_llm.clean_ocr_text(
                                raw_text=raw_text,
                                subject_name=subject_name,
                                year=year,
                                use_advanced=True
                            )
                            if cleaned_text and llm_used != 'none':
                                raw_text = cleaned_text
                                # Note: We don't re-extract questions here to preserve the 
                                # questions_data and images from PyMuPDF structured extraction.
                                # The cleaned text is stored in paper.raw_text for reference.
                                logger.info(f"✓ Text cleaned using {llm_used}")
                            else:
                                logger.warning("LLM cleaning returned no result, using original text")
                        except Exception as e:
                            logger.error(f"LLM text cleaning failed: {e}, using original text")
                    
                    paper.raw_text = raw_text
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
            job.save()
            
            # Delete existing questions for this paper to avoid duplicates/stale data
            existing_question_count = Question.objects.filter(paper=paper).count()
            if existing_question_count > 0:
                job.status_detail = f'Deleting {existing_question_count} old questions...'
                job.save()
                paper.status_detail = f'Deleting {existing_question_count} old questions...'
                paper.progress_percent = max(paper.progress_percent, 65)
                paper.save()
                
                logger.info(f"Deleting {existing_question_count} existing questions for paper {paper.id}")
                Question.objects.filter(paper=paper).delete()
                logger.info(f"✓ Deleted {existing_question_count} old questions")
            
            # Now update status to saving
            job.status_detail = 'Creating question records...'
            job.save()
            paper.progress_percent = max(paper.progress_percent, 67)
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
            # Use the parsed question number if available, not the list index
            raw_qnum = q_data.get('question_number', str(idx + 1))
            # Extract the base numeric part (e.g. '13a' -> 13, '7' -> 7)
            num_match = re.match(r'(\d+)', str(raw_qnum))
            q_number = int(num_match.group(1)) if num_match else (idx + 1)

            if q_number <= 10:
                part = 'A'
                module_num = part_a_modules[q_number - 1]
            elif q_number <= 20:
                part = 'B'
                offset = q_number - 11
                module_num = part_b_modules[offset] if 0 <= offset < len(part_b_modules) else part_b_modules[-1]
            else:
                # Questions beyond 20 — assign based on module_hint if available
                part = q_data.get('part', 'B')
                module_num = q_data.get('module_hint', part_b_modules[-1])

            # Preserve original parsed question number; don't overwrite with index
            if 'question_number' not in q_data or not q_data['question_number']:
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
            from paddleocr import PaddleOCR
            from PIL import Image, ImageOps, ImageEnhance, ImageFilter
            import numpy as np
            paddle_ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)
        except Exception as e:
            logger.warning(f"OCR skipped: PaddleOCR/Pillow not available ({e})")
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

                # Run PaddleOCR on the image
                try:
                    img_array = np.array(img.convert("RGB"))
                    raw = paddle_ocr.ocr(img_array)
                    if raw and raw[0]:
                        lines = [line[1][0] for line in raw[0] if line and len(line) >= 2]
                        best_text = "\n".join(lines)
                    else:
                        best_text = ""
                except Exception as e:
                    logger.warning(f"PaddleOCR error on page {page_idx + 1}: {e}")
                    best_text = ""

                if best_text.strip():
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
        
        # Always apply deterministic text cleaning to OCR output
        full_text = self.text_cleaner.clean(full_text)
        logger.info("Applied deterministic text cleaning to OCR output")
        
        questions = self.fallback_extractor.extract_questions(full_text)
        logger.info(f"OCR extracted {len(questions)} questions")
        return questions, full_text
