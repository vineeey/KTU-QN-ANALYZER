"""
Re-process all papers with improved text extraction and similarity detection.
Run this to fix extraction issues and detect more similar questions.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.papers.models import Paper
from apps.questions.models import Question
from apps.analysis.pipeline import AnalysisPipeline

def reprocess_all_papers():
    """Delete old questions and re-process all papers."""
    
    print("=" * 60)
    print("RE-PROCESSING ALL PAPERS")
    print("=" * 60)
    
    # Get all completed papers
    papers = Paper.objects.filter(status='completed')
    total = papers.count()
    
    print(f"\n📄 Found {total} papers to re-process\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}/{total}] Processing: {paper.title}")
        print(f"Year: {paper.year}, Subject: {paper.subject.name}")
        
        try:
            # Delete old questions for this paper
            old_count = Question.objects.filter(paper=paper).count()
            Question.objects.filter(paper=paper).delete()
            print(f"  ✗ Deleted {old_count} old questions")
            
            # Reset paper status
            paper.status = 'pending'
            paper.progress_percent = 0
            paper.questions_extracted = 0
            paper.questions_classified = 0
            paper.save()
            
            # Re-process with improved extraction
            pipeline = AnalysisPipeline(llm_client=None)
            pipeline.analyze_paper(paper)
            
            # Reload to get updated counts
            paper.refresh_from_db()
            print(f"  ✓ Extracted {paper.questions_extracted} questions")
            print(f"  ✓ Classified {paper.questions_classified} questions")
            print(f"  ✓ Status: {paper.status}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            paper.status = 'failed'
            paper.processing_error = str(e)
            paper.save()
    
    print("\n" + "=" * 60)
    print("✓ RE-PROCESSING COMPLETE")
    print("=" * 60)
    
    # Summary
    completed = Paper.objects.filter(status='completed').count()
    failed = Paper.objects.filter(status='failed').count()
    total_questions = Question.objects.count()
    
    print(f"\nSummary:")
    print(f"  Papers completed: {completed}/{total}")
    print(f"  Papers failed: {failed}/{total}")
    print(f"  Total questions: {total_questions}")
    print(f"\n✓ Now check similarity detection with lowered threshold (0.75)")
    print(f"✓ Run: python manage.py shell")
    print(f"   >>> from apps.analysis.services.similarity_detector import SimilarityDetector")
    print(f"   >>> # Test your questions")

if __name__ == '__main__':
    reprocess_all_papers()
