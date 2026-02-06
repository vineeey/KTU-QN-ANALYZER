"""
Background tasks for analysis using Django-Q2.
"""
from django_q.tasks import async_task
from apps.papers.models import Paper
from apps.subjects.models import Subject


def analyze_paper_task(paper_id: str):
    """
    Background task to analyze a paper.
    Called via Django-Q2.
    """
    from .pipeline import AnalysisPipeline
    
    try:
        paper = Paper.objects.get(id=paper_id)
        paper.status = Paper.ProcessingStatus.PROCESSING
        paper.save()
        
        # Run analysis without LLM for speed (use keyword-based classification)
        # LLM is too slow for real-time processing
        pipeline = AnalysisPipeline(llm_client=None)
        pipeline.analyze_paper(paper)
        
    except Paper.DoesNotExist:
        pass
    except Exception as e:
        paper.status = Paper.ProcessingStatus.FAILED
        paper.processing_error = str(e)
        paper.save()


def analyze_subject_topics_task(subject_id: str):
    """
    Background task to analyze topics for a subject.
    Performs clustering and repetition analysis.
    """
    from apps.analytics.clustering import analyze_subject_topics
    
    try:
        subject = Subject.objects.get(id=subject_id)
        
        # Run topic clustering analysis
        results = analyze_subject_topics(subject)
        
        return results
        
    except Subject.DoesNotExist:
        pass
    except Exception as e:
        import logging
        logging.error(f"Topic analysis failed for subject {subject_id}: {e}")


def queue_paper_analysis(paper: Paper):
    """Queue a paper for background analysis."""
    async_task(
        'apps.analysis.tasks.analyze_paper_task',
        str(paper.id),
        task_name=f'analyze_paper_{paper.id}'
    )


def queue_topic_analysis(subject: Subject):
    """Queue topic clustering analysis for a subject."""
    async_task(
        'apps.analysis.tasks.analyze_subject_topics_task',
        str(subject.id),
        task_name=f'analyze_topics_{subject.id}'
    )
