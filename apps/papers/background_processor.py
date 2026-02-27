"""Background processor for papers - runs synchronously but updates progress."""
import logging
from apps.papers.models import Paper
from apps.analysis.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


def process_subject_papers(subject_id):
    """Process all pending/processing papers for a subject."""
    from apps.subjects.models import Subject
    
    try:
        subject = Subject.objects.get(id=subject_id)
    except Subject.DoesNotExist:
        logger.error(f"Subject {subject_id} not found")
        return
    
    # Get all papers marked for processing
    papers = subject.papers.filter(status=Paper.ProcessingStatus.PROCESSING)
    total = papers.count()
    
    logger.info(f"Processing {total} papers for subject {subject.name}")
    
    for index, paper in enumerate(papers, 1):
        try:
            logger.info(f"Processing paper {index}/{total}: {paper.title}")
            
            # Update status
            paper.status_detail = f'Processing paper {index} of {total}...'
            paper.progress_percent = 5
            paper.save()
            
            # Run analysis pipeline
            pipeline = AnalysisPipeline(llm_client=None)
            pipeline.analyze_paper(paper)
            
            logger.info(f"Successfully processed: {paper.title}")
            
        except Exception as e:
            logger.error(f"Failed to process {paper.title}: {str(e)}")
            paper.status = Paper.ProcessingStatus.FAILED
            paper.processing_error = str(e)
            paper.status_detail = f'Error: {str(e)}'
            paper.save()
    
    logger.info(f"Completed processing {total} papers for subject {subject.name}")
