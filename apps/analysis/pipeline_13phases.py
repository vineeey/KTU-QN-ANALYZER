"""
13-PHASE PIPELINE IMPLEMENTATION

This module implements the EXACT workflow specified in copilot-instructions.md.
Each phase is isolated, testable, and follows deterministic rules where possible.

ARCHITECTURE:
• NO ML logic in views
• Pure functions where possible
• Job-scoped processing
• Clear phase separation
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import uuid
from django.utils import timezone
from django.db import transaction

from apps.analysis.models import AnalysisJob
from apps.analysis.job_models import TempPaper, TempQuestion, TempTopicCluster

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: USER UPLOAD
# ═══════════════════════════════════════════════════════════════

class Phase1_Upload:
    """
    PHASE 1: Accept multiple PDFs and create job workspace.
    """
    
    @staticmethod
    def create_job(subject_name: str, pdf_files: List) -> AnalysisJob:
        """
        Create analysis job and workspace.
        
        Args:
            subject_name: User-provided subject name
            pdf_files: List of uploaded PDF files
        
        Returns:
            AnalysisJob instance with UUID
        """
        job = AnalysisJob.objects.create(
            subject_name=subject_name,
            pdf_count=len(pdf_files),
            status=AnalysisJob.Status.UPLOADING
        )
        
        # Create workspace directory
        workspace = Path(f"media/jobs/{job.id}")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "pdfs").mkdir(exist_ok=True)
        (workspace / "output").mkdir(exist_ok=True)
        
        logger.info(f"Created job {job.id} with workspace at {workspace}")
        return job
    
    @staticmethod
    def validate_and_save_pdfs(job: AnalysisJob, pdf_files: List) -> List[TempPaper]:
        """
        Validate and save uploaded PDFs.
        
        Args:
            job: AnalysisJob instance
            pdf_files: List of InMemoryUploadedFile instances
        
        Returns:
            List of TempPaper instances
        """
        papers = []
        
        for pdf_file in pdf_files:
            # Validate file type
            if not pdf_file.name.lower().endswith('.pdf'):
                raise ValueError(f"Invalid file type: {pdf_file.name}")
            
            # Validate size (max 50MB)
            if pdf_file.size > 50 * 1024 * 1024:
                raise ValueError(f"File too large: {pdf_file.name}")
            
            # Create TempPaper
            paper = TempPaper.objects.create(
                job=job,
                file=pdf_file,
                filename=pdf_file.name,
                file_size=pdf_file.size
            )
            papers.append(paper)
        
        logger.info(f"Saved {len(papers)} PDFs for job {job.id}")
        return papers


# ═══════════════════════════════════════════════════════════════
# PHASE 2: PDF TYPE DETECTION
# ═══════════════════════════════════════════════════════════════

class Phase2_PDFDetection:
    """
    PHASE 2: Detect PDF type and extract raw text.
    Uses pdfplumber for text-based, PyMuPDF+OCR for scanned.
    """
    
    @staticmethod
    def detect_and_extract(paper: TempPaper) -> str:
        """
        Detect PDF type and extract text.
        
        Args:
            paper: TempPaper instance
        
        Returns:
            Extracted raw text (unchanged)
        """
        from apps.analysis.services.pdf_extractor import PDFExtractor
        import re
        
        extractor = PDFExtractor()
        text = extractor.extract_text(paper.file.path)
        
        # Extract year from filename (e.g., "2024.pdf", "S3_2023_May.pdf")
        year_match = re.search(r'(20\d{2})', paper.filename)
        if year_match:
            paper.year = year_match.group(1)
        
        # Simple heuristic for PDF type detection
        paper.pdf_type = 'text' if len(text) > 100 else 'scanned'
        paper.raw_text = text
        paper.page_count = len(text) // 2000 + 1  # Rough estimate
        paper.extracted = True
        paper.processed_at = timezone.now()
        paper.save()
        
        logger.info(f"Extracted {len(text)} chars from {paper.filename} ({paper.pdf_type})")
        return text


# ═══════════════════════════════════════════════════════════════
# PHASE 3: QUESTION SEGMENTATION (RULE-BASED)
# ═══════════════════════════════════════════════════════════════

class Phase3_QuestionSegmentation:
    """
    PHASE 3: Extract logical questions using KTU pattern rules.
    
    Rules:
    - Detect PART A and PART B sections
    - Extract question number, text, marks
    - Handle OR questions and sub-parts
    - Each logical question = one semantic unit
    """
    
    @staticmethod
    def segment_questions(paper: TempPaper) -> List[TempQuestion]:
        """
        Segment paper into logical questions.
        
        Args:
            paper: TempPaper with extracted text
        
        Returns:
            List of TempQuestion instances
        """
        from apps.analysis.services.extractor import QuestionExtractor
        
        extractor = QuestionExtractor()
        questions_data = extractor.extract_questions(paper.raw_text)
        
        questions = []
        for q_data in questions_data:
            question = TempQuestion.objects.create(
                paper=paper,
                question_number=q_data['question_number'],
                part=q_data['part'],
                marks=q_data['marks'],
                raw_text=q_data['text'],
                has_sub_parts=q_data.get('has_sub_parts', False),
                sub_parts=q_data.get('sub_parts', []),
                is_or_question=q_data.get('is_or_question', False),
                has_images=q_data.get('has_images', False),
                images=q_data.get('images', [])
            )
            questions.append(question)
        
        logger.info(f"Segmented {len(questions)} questions from {paper.filename}")
        return questions


# ═══════════════════════════════════════════════════════════════
# PHASE 4: MODULE MAPPING (RULE-BASED)
# ═══════════════════════════════════════════════════════════════

class Phase4_ModuleMapping:
    """
    PHASE 4: Map questions to modules using KTU fixed rules.
    
    KTU 2019 Scheme Mapping:
    Q1-2, Q11-12   → Module 1
    Q3-4, Q13-14   → Module 2
    Q5-6, Q15-16   → Module 3
    Q7-8, Q17-18   → Module 4
    Q9-10, Q19-20  → Module 5
    """
    
    KTU_MODULE_MAPPING = {
        1: 1, 2: 1, 11: 1, 12: 1,   # Module 1
        3: 2, 4: 2, 13: 2, 14: 2,   # Module 2
        5: 3, 6: 3, 15: 3, 16: 3,   # Module 3
        7: 4, 8: 4, 17: 4, 18: 4,   # Module 4
        9: 5, 10: 5, 19: 5, 20: 5,  # Module 5
    }
    
    @staticmethod
    def map_question_to_module(question: TempQuestion) -> int:
        """
        Map question to module number.
        
        Args:
            question: TempQuestion instance
        
        Returns:
            Module number (1-5)
        """
        try:
            # Extract base question number (e.g., "11a" → 11)
            base_num = int(''.join(filter(str.isdigit, question.question_number)))
            module = Phase4_ModuleMapping.KTU_MODULE_MAPPING.get(base_num)
            
            if module:
                question.module_number = module
                question.save(update_fields=['module_number'])
                return module
            else:
                logger.warning(f"No module mapping for question {question.question_number}")
                return None
        except (ValueError, TypeError):
            logger.error(f"Invalid question number: {question.question_number}")
            return None
    
    @staticmethod
    def map_all_questions(job: AnalysisJob):
        """Map all questions in job to modules."""
        questions = TempQuestion.objects.filter(paper__job=job)
        mapped_count = 0
        
        for question in questions:
            if Phase4_ModuleMapping.map_question_to_module(question):
                mapped_count += 1
        
        logger.info(f"Mapped {mapped_count}/{questions.count()} questions to modules")


# ═══════════════════════════════════════════════════════════════
# PHASE 5: QUESTION NORMALIZATION
# ═══════════════════════════════════════════════════════════════

class Phase5_Normalization:
    """
    PHASE 5: Normalize questions for semantic analysis.
    
    Rules:
    - Remove question numbers
    - Remove marks references
    - Remove year references
    - Preserve academic meaning
    - DO NOT overwrite raw_text
    """
    
    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """
        Normalize question text for embedding.
        
        Args:
            raw_text: Original question text
        
        Returns:
            Normalized text
        """
        import re
        
        text = raw_text
        
        # Remove question numbers (e.g., "1.", "11a)", "12(i)")
        text = re.sub(r'^\s*\d+[a-z]?[\.\)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*[\(\[]?[ivxlc]+[\)\]]?\s*', '', text, flags=re.MULTILINE)
        
        # Remove marks references (e.g., "(3 marks)", "[14]")
        text = re.sub(r'\(?\d+\s*marks?\)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove OR indicators (but keep the question text)
        text = re.sub(r'\bOR\b', '', text)
        
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def normalize_all_questions(job: AnalysisJob):
        """Normalize all questions in job."""
        questions = TempQuestion.objects.filter(paper__job=job)
        
        for question in questions:
            question.normalized_text = Phase5_Normalization.normalize_text(question.raw_text)
            question.save(update_fields=['normalized_text'])
        
        logger.info(f"Normalized {questions.count()} questions")


# ═══════════════════════════════════════════════════════════════
# PHASE 6: EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════

class Phase6_Embeddings:
    """
    PHASE 6: Generate embeddings using sentence-transformers.
    
    Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    Process: Module-wise, combining Part A + Part B
    """
    
    @staticmethod
    def generate_embeddings(job: AnalysisJob):
        """
        Generate embeddings for all questions.
        
        Uses local sentence-transformer model (no API calls).
        """
        from apps.analysis.services.embedder import EmbeddingService
        
        embedder = EmbeddingService()
        
        # Process each module separately
        for module_num in range(1, 6):
            questions = TempQuestion.objects.filter(
                paper__job=job,
                module_number=module_num
            )
            
            if not questions.exists():
                continue
            
            # Get normalized texts
            texts = [q.normalized_text for q in questions]
            
            # Generate embeddings in batch
            embeddings = embedder.embed_batch(texts)
            
            # Save embeddings
            for question, embedding in zip(questions, embeddings):
                question.embedding = embedding.tolist()  # Convert numpy to list
                question.save(update_fields=['embedding'])
            
            logger.info(f"Generated embeddings for {len(texts)} questions in Module {module_num}")


# ═══════════════════════════════════════════════════════════════
# PHASE 7: TOPIC CLUSTERING
# ═══════════════════════════════════════════════════════════════

class Phase7_Clustering:
    """
    PHASE 7: Cluster questions into topics using HDBSCAN.
    
    Rules:
    - Cluster PER MODULE
    - Each cluster = one exam topic
    - Noise questions allowed (cluster_id = -1)
    """
    
    @staticmethod
    def cluster_module(job: AnalysisJob, module_num: int) -> List[TempTopicCluster]:
        """
        Cluster questions in one module.
        
        Args:
            job: AnalysisJob instance
            module_num: Module number (1-5)
        
        Returns:
            List of TempTopicCluster instances
        """
        from apps.analysis.services.clustering import QuestionClusterer
        import numpy as np
        
        questions = TempQuestion.objects.filter(
            paper__job=job,
            module_number=module_num,
            embedding__isnull=False
        )
        
        if questions.count() < 2:
            logger.info(f"Module {module_num}: Too few questions for clustering")
            return []
        
        # Get embeddings
        embeddings = np.array([q.embedding for q in questions])
        
        # Perform clustering using HDBSCAN
        clusterer = QuestionClusterer()
        
        # Use HDBSCAN clustering method
        try:
            import hdbscan
            clusterer_model = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            cluster_labels = clusterer_model.fit_predict(embeddings)
        except ImportError:
            logger.warning("HDBSCAN not available, falling back to agglomerative clustering")
            from sklearn.cluster import AgglomerativeClustering
            clusterer_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='euclidean',
                linkage='average'
            )
            cluster_labels = clusterer_model.fit_predict(embeddings)
        
        # Assign cluster IDs to questions
        for question, cluster_id in zip(questions, cluster_labels):
            question.cluster_id = int(cluster_id)
            question.save(update_fields=['cluster_id'])
        
        # Create TopicCluster objects
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Exclude noise
        
        clusters = []
        for cluster_id in unique_clusters:
            cluster_questions = [q for q, label in zip(questions, cluster_labels) if label == cluster_id]
            
            # Get representative question (most central)
            representative = cluster_questions[0]  # Simplified - could use centroid
            
            cluster = TempTopicCluster.objects.create(
                job=job,
                module_number=module_num,
                cluster_id=int(cluster_id),
                topic_label=Phase7_Clustering._generate_topic_label(cluster_questions),
                representative_question=representative.raw_text[:200]
            )
            
            # Link questions to cluster
            for q in cluster_questions:
                q.topic_cluster = cluster
                q.save(update_fields=['topic_cluster'])
            
            clusters.append(cluster)
        
        logger.info(f"Module {module_num}: Created {len(clusters)} topic clusters")
        return clusters
    
    @staticmethod
    def _generate_topic_label(questions: List[TempQuestion]) -> str:
        """Generate human-readable topic label from cluster questions."""
        # Simplified - could use keyword extraction or LLM
        # For now, use first question as label
        if questions:
            return questions[0].normalized_text[:100] + "..."
        return "Unnamed Topic"


# ═══════════════════════════════════════════════════════════════
# PHASE 8-10: PRIORITY SCORING
# ═══════════════════════════════════════════════════════════════

class Phase8_PriorityScoring:
    """
    PHASE 8-10: Compute priority scores and assign tiers.
    
    Metrics:
    - Frequency (distinct years)
    - Average marks
    - Part A vs Part B contribution (NEW)
    - Confidence score (NEW)
    
    Formula: Priority Score = (2 × Frequency) + Average Marks
    """
    
    @staticmethod
    def compute_all_scores(job: AnalysisJob):
        """Compute priority scores for all topic clusters."""
        # Get total years in job
        total_years = job.total_years
        
        clusters = TempTopicCluster.objects.filter(job=job)
        
        for cluster in clusters:
            cluster.calculate_metrics(total_years_in_job=total_years)
        
        logger.info(f"Computed priority scores for {clusters.count()} clusters")


# ═══════════════════════════════════════════════════════════════
# PHASE 11: MODULE-WISE PDF GENERATION
# ═══════════════════════════════════════════════════════════════

class Phase11_PDFGeneration:
    """
    PHASE 11: Generate module-wise PDFs.
    
    Each PDF contains:
    - SECTION A: Complete question bank (Part A + Part B, year-wise)
    - SECTION B: Repeated question analysis (tier-wise)
    - FINAL STUDY PRIORITY ORDER
    """
    
    @staticmethod
    def generate_module_pdf(job: AnalysisJob, module_num: int) -> str:
        """
        Generate PDF for one module.
        
        Args:
            job: AnalysisJob instance
            module_num: Module number (1-5)
        
        Returns:
            Path to generated PDF
        """
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.colors import HexColor
        from collections import defaultdict
        
        # Get data
        questions = TempQuestion.objects.filter(
            paper__job=job,
            module_number=module_num
        ).select_related('paper').order_by('paper__year', 'part', 'question_number')
        
        clusters = TempTopicCluster.objects.filter(
            job=job,
            module_number=module_num
        ).order_by('priority_tier', '-priority_score')
        
        # Output path
        output_path = f"media/jobs/{job.id}/output/Module_{module_num}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#667eea'),
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph(f"{job.subject_name}", title_style))
        story.append(Paragraph(f"Module {module_num} - Question Bank & Priority Analysis", styles['Heading2']))
        story.append(Spacer(1, 0.5*cm))
        
        # Section A: Complete Question Bank
        story.append(Paragraph("SECTION A: COMPLETE QUESTION BANK", styles['Heading2']))
        story.append(Spacer(1, 0.3*cm))
        
        # Group by year and part
        questions_by_year = defaultdict(lambda: {'A': [], 'B': []})
        for q in questions:
            year = q.get_year() or 'Unknown'
            questions_by_year[year][q.part].append(q)
        
        for year in sorted(questions_by_year.keys(), reverse=True):
            story.append(Paragraph(f"<b>Year: {year}</b>", styles['Heading3']))
            
            # Part A
            if questions_by_year[year]['A']:
                story.append(Paragraph("Part A (Short Answer)", styles['Heading4']))
                for q in questions_by_year[year]['A']:
                    story.append(Paragraph(f"Q{q.question_number}. {q.raw_text} ({q.marks} marks)", styles['Normal']))
                    story.append(Spacer(1, 0.2*cm))
            
            # Part B
            if questions_by_year[year]['B']:
                story.append(Paragraph("Part B (Essay)", styles['Heading4']))
                for q in questions_by_year[year]['B']:
                    story.append(Paragraph(f"Q{q.question_number}. {q.raw_text} ({q.marks} marks)", styles['Normal']))
                    story.append(Spacer(1, 0.2*cm))
            
            story.append(Spacer(1, 0.5*cm))
        
        story.append(PageBreak())
        
        # Section B: Priority Analysis
        story.append(Paragraph("SECTION B: REPEATED QUESTION ANALYSIS", styles['Heading2']))
        story.append(Spacer(1, 0.3*cm))
        
        # Group by tier
        clusters_by_tier = defaultdict(list)
        for cluster in clusters:
            clusters_by_tier[cluster.priority_tier].append(cluster)
        
        tier_names = {
            1: 'Tier 1 - Very High Priority',
            2: 'Tier 2 - High Priority',
            3: 'Tier 3 - Medium Priority',
            4: 'Tier 4 - Low Priority'
        }
        
        for tier in [1, 2, 3, 4]:
            if tier in clusters_by_tier:
                story.append(Paragraph(tier_names[tier], styles['Heading3']))
                for cluster in clusters_by_tier[tier]:
                    story.append(Paragraph(f"<b>{cluster.topic_label}</b>", styles['Heading4']))
                    story.append(Paragraph(f"Frequency: {cluster.frequency} years", styles['Normal']))
                    story.append(Paragraph(f"Confidence: {cluster.confidence_score:.1f}%", styles['Normal']))
                    story.append(Paragraph(f"Part A: {cluster.part_a_count} | Part B: {cluster.part_b_count}", styles['Normal']))
                    story.append(Paragraph(f"Years: {', '.join(map(str, cluster.years_appeared))}", styles['Normal']))
                    story.append(Spacer(1, 0.3*cm))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated PDF for Module {module_num}: {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════
# PHASE 13: AUTO CLEANUP
# ═══════════════════════════════════════════════════════════════

class Phase13_Cleanup:
    """
    PHASE 13: Auto-cleanup expired jobs.
    
    Deletes:
    - All temporary models (TempPaper, TempQuestion, TempTopicCluster)
    - All uploaded files
    - Job workspace directory
    """
    
    @staticmethod
    def cleanup_job(job: AnalysisJob):
        """
        Clean up all data for a job.
        
        Args:
            job: AnalysisJob instance
        """
        import shutil
        
        # Delete workspace directory
        workspace = Path(f"media/jobs/{job.id}")
        if workspace.exists():
            shutil.rmtree(workspace)
            logger.info(f"Deleted workspace: {workspace}")
        
        # Delete job (cascade deletes all related models)
        job_id = job.id
        job.delete()
        logger.info(f"Deleted job {job_id}")
    
    @staticmethod
    def cleanup_expired_jobs():
        """Clean up all expired jobs (cron task)."""
        expired_jobs = AnalysisJob.objects.filter(
            expires_at__lt=timezone.now(),
            status__in=[AnalysisJob.Status.COMPLETED, AnalysisJob.Status.FAILED]
        )
        
        count = expired_jobs.count()
        for job in expired_jobs:
            Phase13_Cleanup.cleanup_job(job)
        
        logger.info(f"Cleaned up {count} expired jobs")
        return count


# ═══════════════════════════════════════════════════════════════
# COMPLETE PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class CompletePipeline:
    """
    Orchestrates the complete 13-phase pipeline.
    
    This is called from views (views only orchestrate, no ML logic).
    """
    
    @staticmethod
    def run_complete_analysis(job_id: uuid.UUID):
        """
        Run complete 13-phase analysis for a job.
        
        Args:
            job_id: AnalysisJob UUID
        """
        job = AnalysisJob.objects.get(id=job_id)
        
        try:
            # PHASE 2: PDF Detection & Extraction
            job.status = AnalysisJob.Status.DETECTING_PDF_TYPE
            job.save()
            papers = TempPaper.objects.filter(job=job)
            for paper in papers:
                Phase2_PDFDetection.detect_and_extract(paper)
            
            # Extract years from papers
            years = sorted(set(p.year for p in papers if p.year))
            job.years_list = years
            job.total_years = len(years)
            job.save()
            
            # PHASE 3: Question Segmentation
            job.status = AnalysisJob.Status.EXTRACTING
            job.save()
            for paper in papers:
                Phase3_QuestionSegmentation.segment_questions(paper)
            
            job.questions_extracted = TempQuestion.objects.filter(paper__job=job).count()
            job.save()
            
            # PHASE 4: Module Mapping
            job.status = AnalysisJob.Status.MAPPING_MODULES
            job.save()
            Phase4_ModuleMapping.map_all_questions(job)
            
            # PHASE 5: Normalization
            job.status = AnalysisJob.Status.NORMALIZING
            job.save()
            Phase5_Normalization.normalize_all_questions(job)
            
            # PHASE 6: Embeddings
            job.status = AnalysisJob.Status.EMBEDDING
            job.save()
            Phase6_Embeddings.generate_embeddings(job)
            
            # PHASE 7: Clustering
            job.status = AnalysisJob.Status.CLUSTERING
            job.save()
            for module_num in range(1, 6):
                Phase7_Clustering.cluster_module(job, module_num)
            
            # PHASE 8-10: Priority Scoring
            job.status = AnalysisJob.Status.SCORING
            job.save()
            Phase8_PriorityScoring.compute_all_scores(job)
            
            job.topics_clustered = TempTopicCluster.objects.filter(job=job).count()
            job.save()
            
            # PHASE 11: PDF Generation
            job.status = AnalysisJob.Status.GENERATING_PDFS
            job.save()
            output_pdfs = {}
            for module_num in range(1, 6):
                pdf_path = Phase11_PDFGeneration.generate_module_pdf(job, module_num)
                output_pdfs[f'module_{module_num}'] = pdf_path
            
            job.output_pdfs = output_pdfs
            job.save()
            
            # Mark completed
            job.mark_completed()
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            job.mark_failed(str(e))
