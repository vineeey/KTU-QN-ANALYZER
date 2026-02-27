"""
Complete analysis pipeline integrating all services.
Implements exact workflow from master prompt.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from django.utils import timezone

from apps.papers.models import Paper
from apps.questions.models import Question
from apps.subjects.models import Module
from .models import AnalysisJob
from .services.pdf_extractor import PDFExtractor, QuestionSegmenter
from .services.embedder import EmbeddingService
from .services.similarity_detector import SimilarityDetector
from .services.clustering import QuestionClusterer
from apps.reports.module_report_generator_v2 import ModuleReportGenerator

logger = logging.getLogger(__name__)


class CompletePipeline:
    """
    Complete PYQ analysis pipeline following master prompt specification.
    
    Workflow:
    1. PDF Extraction (pdfplumber → PyMuPDF → OCR)
    2. Question Segmentation (regex)
    3. Module Mapping (deterministic rules)
    4. Embedding Generation (all-MiniLM-L6-v2)
    5. Similarity Detection (cosine similarity)
    6. Clustering (Agglomerative/HDBSCAN)
    7. Priority Assignment (4-tier frequency-based)
    8. PDF Generation (WeasyPrint + Jinja2)
    """
    
    def __init__(self):
        # Core services
        self.pdf_extractor = PDFExtractor()
        self.segmenter = QuestionSegmenter()
        self.embedder = EmbeddingService()
        self.similarity_detector = SimilarityDetector(threshold=0.85)
        self.clusterer = QuestionClusterer(similarity_threshold=0.85)
    
    def analyze_paper(self, paper: Paper) -> AnalysisJob:
        """
        Run complete analysis on a paper.
        
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
            logger.info(f"Starting complete analysis for {paper.title}")
            
            # STEP 1: PDF EXTRACTION
            job.status = 'extracting'
            job.progress = 10
            job.status_detail = 'Extracting text from PDF...'
            job.save()
            
            questions = self.pdf_extractor.extract_questions_with_metadata(
                paper.file.path,
                year=paper.year,
                session=paper.session or 'Unknown'
            )
            
            logger.info(f"Extracted {len(questions)} questions")
            
            job.questions_extracted = len(questions)
            job.progress = 30
            job.save()
            
            # STEP 2: MODULE MAPPING
            job.status = 'classifying'
            job.progress = 40
            job.status_detail = 'Mapping questions to modules...'
            job.save()
            
            questions_with_modules = self._map_to_modules(questions, paper.subject)
            
            job.progress = 50
            job.save()
            
            # STEP 3: SAVE TO DATABASE
            job.status = 'analyzing'
            job.progress = 60
            job.status_detail = 'Saving questions to database...'
            job.save()
            
            created_questions = self._save_questions(questions_with_modules, paper)
            
            job.progress = 70
            job.save()
            
            # STEP 4: EMBEDDING & CLUSTERING (per module)
            job.status = 'clustering'
            job.progress = 75
            job.status_detail = 'Analyzing repeated questions...'
            job.save()
            
            self._analyze_and_cluster_by_module(paper.subject)
            
            job.progress = 90
            job.save()
            
            # STEP 5: GENERATE MODULE REPORTS
            job.status = 'generating'
            job.progress = 95
            job.status_detail = 'Generating module PDFs...'
            job.save()
            
            report_paths = self._generate_all_module_reports(paper.subject)
            
            # Complete
            job.status = 'completed'
            job.progress = 100
            job.status_detail = f'Analysis complete. Generated {len(report_paths)} module reports.'
            job.completed_at = timezone.now()
            job.save()
            
            paper.status = 'completed'
            paper.progress_percent = 100
            paper.save()
            
            logger.info(f"✓ Analysis complete for {paper.title}")
            return job
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            job.status = 'failed'
            job.status_detail = str(e)
            job.save()
            
            paper.status = 'failed'
            paper.status_detail = str(e)
            paper.save()
            
            raise
    
    def _map_to_modules(self, questions: List[Dict[str, Any]], subject) -> List[Dict[str, Any]]:
        """
        Map questions to modules using deterministic rules.
        
        KTU Mapping:
        - PART A: Qn 1-2 → Module 1, Qn 3-4 → Module 2, etc.
        - PART B: Qn 11-12 → Module 1, Qn 13-14 → Module 2, etc.
        """
        modules = list(subject.modules.all().order_by('number'))
        
        for q in questions:
            qn = q.get('question_number', 0)
            part = q.get('part', 'A')
            
            if part == 'A':
                # PART A: 1-2, 3-4, 5-6, 7-8, 9-10
                module_num = ((qn - 1) // 2) + 1
            else:  # PART B
                # PART B: 11-12, 13-14, 15-16, 17-18, 19-20
                module_num = ((qn - 11) // 2) + 1
            
            # Clamp to valid range
            module_num = max(1, min(5, module_num))
            
            # Find module
            module = next((m for m in modules if m.number == module_num), None)
            q['module'] = module
            q['module_number'] = module_num
        
        return questions
    
    def _save_questions(self, questions: List[Dict[str, Any]], paper: Paper) -> List[Question]:
        """Save questions to database."""
        created = []
        
        for q_data in questions:
            question = Question.objects.create(
                paper=paper,
                text=q_data['text'],
                question_number=q_data.get('question_number'),
                part=q_data.get('part', 'A'),
                marks=q_data.get('marks', 0),
                module=q_data.get('module'),
                year=q_data.get('year'),
                session=q_data.get('session', '')
            )
            created.append(question)
        
        logger.info(f"Saved {len(created)} questions to database")
        return created
    
    def _analyze_and_cluster_by_module(self, subject):
        """
        Analyze and cluster questions for each module.
        Generates embeddings and identifies repeated questions.
        """
        modules = subject.modules.all()
        
        for module in modules:
            logger.info(f"Analyzing Module {module.number}...")
            
            # Get all questions for this module
            questions = Question.objects.filter(
                module=module,
                paper__subject=subject
            ).values('id', 'text', 'year', 'session', 'part', 'marks')
            
            if not questions:
                continue
            
            questions_list = list(questions)
            
            # Generate embeddings
            texts = [q['text'] for q in questions_list]
            embeddings = self.embedder.embed_batch(texts)
            
            # Cluster similar questions
            clusters = self.clusterer.cluster_agglomerative(embeddings, questions_list)
            
            # Save clusters to database (for later report generation)
            self._save_clusters(clusters, module, subject)
            
            logger.info(f"Module {module.number}: Found {len(clusters)} topic clusters")
    
    def _save_clusters(self, clusters: List[Dict[str, Any]], module, subject):
        """Save clusters to database for reporting."""
        from apps.analytics.models import TopicCluster
        
        # Clear existing clusters for this module
        TopicCluster.objects.filter(module=module, subject=subject).delete()
        
        # Save new clusters
        for cluster_data in clusters:
            TopicCluster.objects.create(
                subject=subject,
                module=module,
                topic_name=cluster_data['topic_name'],
                representative_text=cluster_data['representative_text'],
                frequency_count=cluster_data['frequency'],
                priority_tier=cluster_data['tier'],
                years_appeared=cluster_data['years'],
                cluster_id=cluster_data['cluster_id']
            )
    
    def _generate_all_module_reports(self, subject) -> List[str]:
        """Generate PDF reports for all 5 modules."""
        report_paths = []
        
        for module_num in range(1, 6):
            module = Module.objects.filter(subject=subject, number=module_num).first()
            
            if not module:
                logger.warning(f"Module {module_num} not found, skipping")
                continue
            
            # Get questions and clusters
            questions = Question.objects.filter(
                module=module,
                paper__subject=subject
            ).values('text', 'year', 'session', 'part', 'marks', 'question_number')
            
            from apps.analytics.models import TopicCluster
            clusters_data = TopicCluster.objects.filter(
                module=module,
                subject=subject
            ).order_by('-frequency_count')
            
            # Convert to dict format
            questions_list = list(questions)
            clusters_list = [
                {
                    'topic_name': c.topic_name,
                    'frequency': c.frequency_count,
                    'tier': c.priority_tier,
                    'years': c.years_appeared,
                    'representative_text': c.representative_text
                }
                for c in clusters_data
            ]
            
            # Generate PDF
            generator = ModuleReportGenerator(subject, module_num)
            pdf_path = generator.generate(questions_list, clusters_list)
            
            if pdf_path:
                report_paths.append(pdf_path)
        
        logger.info(f"Generated {len(report_paths)} module reports")
        return report_paths


def analyze_paper_complete(paper: Paper) -> AnalysisJob:
    """
    Convenience function to run complete analysis.
    
    Usage:
        from apps.analysis.pipeline_complete import analyze_paper_complete
        job = analyze_paper_complete(paper)
    """
    pipeline = CompletePipeline()
    return pipeline.analyze_paper(paper)
