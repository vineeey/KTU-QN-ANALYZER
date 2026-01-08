"""
MIGRATION GUIDE - Transitioning to Refactored Architecture

This guide helps you migrate from the old authentication-based system
to the new guest-workflow job-based system.
"""

# ═══════════════════════════════════════════════════════════════
# STEP 1: Backup Current Database
# ═══════════════════════════════════════════════════════════════

"""
Before running any migrations, backup your current database:

    cp db/pyq_analyzer.sqlite3 db/pyq_analyzer.sqlite3.backup
    
If something goes wrong, you can restore:

    cp db/pyq_analyzer.sqlite3.backup db/pyq_analyzer.sqlite3
"""


# ═══════════════════════════════════════════════════════════════
# STEP 2: Create Migrations for New Models
# ═══════════════════════════════════════════════════════════════

"""
The new models need to be registered with Django:

1. Update apps/analysis/__init__.py to import new models:
"""

# apps/analysis/__init__.py
from apps.analysis.job_models import TempPaper, TempQuestion, TempTopicCluster

"""
2. Update apps/analysis/models.py to include new models in __all__:
"""

# apps/analysis/models.py
__all__ = ['AnalysisJob']

"""
3. Create migrations:
"""

# Terminal commands:
"""
python manage.py makemigrations analysis --name "refactor_to_job_based_architecture"
python manage.py migrate analysis
"""


# ═══════════════════════════════════════════════════════════════
# STEP 3: Optional - Migrate Existing Data
# ═══════════════════════════════════════════════════════════════

"""
If you want to preserve existing analysis data as guest jobs:
"""

from django.core.management.base import BaseCommand
from apps.papers.models import Paper
from apps.analysis.models import AnalysisJob
from apps.analysis.job_models import TempPaper, TempQuestion
from django.utils import timezone

class MigrateToGuestJobs(BaseCommand):
    """
    One-time migration script to convert existing papers to guest jobs.
    
    WARNING: This creates NEW job-based data. Old data is preserved.
    """
    
    def handle(self, *args, **options):
        papers = Paper.objects.all()
        
        for paper in papers:
            # Create guest job for this paper
            job = AnalysisJob.objects.create(
                subject_name=paper.subject.name,
                pdf_count=1,
                total_years=1,
                years_list=[paper.year] if paper.year else [],
                status=AnalysisJob.Status.COMPLETED,
                completed_at=paper.processed_at or timezone.now()
            )
            
            # Create TempPaper
            temp_paper = TempPaper.objects.create(
                job=job,
                file=paper.file,
                filename=paper.title,
                year=paper.year,
                raw_text=paper.raw_text,
                extracted=True
            )
            
            # Migrate questions
            for question in paper.questions.all():
                TempQuestion.objects.create(
                    paper=temp_paper,
                    question_number=question.question_number,
                    part='A' if question.marks <= 3 else 'B',
                    marks=question.marks or 3,
                    raw_text=question.text,
                    normalized_text=question.text,
                    module_number=question.module_id if question.module else None
                )
            
            self.stdout.write(f"Migrated paper: {paper.title} → Job {job.id}")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Update Settings
# ═══════════════════════════════════════════════════════════════

"""
No changes needed to settings.py - the authentication system is kept
as OPTIONAL. Users can still register if they want, but it's not required.
"""


# ═══════════════════════════════════════════════════════════════
# STEP 5: Test Guest Upload Flow
# ═══════════════════════════════════════════════════════════════

"""
1. Start server:
    python manage.py runserver

2. Open browser:
    http://localhost:8000/

3. You should see:
    - Guest upload form
    - No login/register buttons (unless you add them back)
    - Clean, modern UI

4. Test upload:
    - Enter subject name: "Test Subject"
    - Upload 1-2 PDF files
    - Submit form
    - Should redirect to processing page
    - Should show progress updates

5. Test download:
    - Wait for completion
    - Should see 5 module download buttons
    - Click to download PDFs
"""


# ═══════════════════════════════════════════════════════════════
# STEP 6: Set Up Auto-Cleanup
# ═══════════════════════════════════════════════════════════════

"""
1. Create cron job (production):
"""

# crontab -e
"""
# Add this line:
0 * * * * cd /path/to/KTU-QN-ANALYZER && /path/to/python manage.py cleanup_expired_jobs >> /var/log/ktu-cleanup.log 2>&1
"""

"""
2. OR use Django-Q (recommended):
"""

# In Django admin or shell:
from django_q.models import Schedule
from django_q.tasks import schedule

schedule(
    'django.core.management.call_command',
    'cleanup_expired_jobs',
    schedule_type=Schedule.HOURLY,
    name='Cleanup Expired Jobs'
)


# ═══════════════════════════════════════════════════════════════
# STEP 7: Verify All Phases Work
# ═══════════════════════════════════════════════════════════════

"""
Test individual phases in Django shell:
"""

# python manage.py shell

from apps.analysis.models import AnalysisJob
from apps.analysis.pipeline_13phases import *

# Test job creation
job = AnalysisJob.objects.create(
    subject_name="Test Subject",
    pdf_count=1,
    total_years=1,
    years_list=["2024"]
)

# Test PDF detection (requires actual PDF file)
# from apps.analysis.job_models import TempPaper
# paper = TempPaper.objects.create(job=job, file="path/to/test.pdf")
# Phase2_PDFDetection.detect_and_extract(paper)

# Test module mapping
from apps.analysis.job_models import TempQuestion
question = TempQuestion.objects.create(
    paper=paper,
    question_number="11",
    part="B",
    marks=14,
    raw_text="Test question text"
)
module = Phase4_ModuleMapping.map_question_to_module(question)
print(f"Question 11 mapped to Module {module}")  # Should be 1

# Test normalization
Phase5_Normalization.normalize_all_questions(job)
question.refresh_from_db()
print(f"Normalized: {question.normalized_text}")

# Test cleanup
Phase13_Cleanup.cleanup_job(job)
print("Job cleaned up successfully")


# ═══════════════════════════════════════════════════════════════
# STEP 8: Rollback Plan (If Needed)
# ═══════════════════════════════════════════════════════════════

"""
If you need to rollback to the old system:

1. Restore database:
    cp db/pyq_analyzer.sqlite3.backup db/pyq_analyzer.sqlite3

2. Revert URL changes in config/urls.py:
    path('', include('apps.core.urls')),  # Old home page

3. Revert migration:
    python manage.py migrate analysis <previous_migration_name>

4. Delete new files:
    rm apps/analysis/job_models.py
    rm apps/analysis/pipeline_13phases.py
    rm apps/core/guest_views.py
"""


# ═══════════════════════════════════════════════════════════════
# STEP 9: Production Deployment
# ═══════════════════════════════════════════════════════════════

"""
1. Update requirements.txt:
    pip freeze > requirements.txt

2. Run migrations on production:
    python manage.py migrate

3. Collect static files:
    python manage.py collectstatic --noinput

4. Restart web server:
    systemctl restart gunicorn  # or your server

5. Set up cron job on production server

6. Monitor logs:
    tail -f /var/log/ktu-analyzer.log
"""


# ═══════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════

"""
Issue: ImportError for job_models

Solution:
    Make sure apps/analysis/job_models.py is in the correct location
    Check apps/analysis/__init__.py imports

Issue: Migration conflicts

Solution:
    python manage.py makemigrations --merge
    python manage.py migrate

Issue: Media files not serving

Solution:
    Check MEDIA_ROOT and MEDIA_URL in settings.py
    Verify media/jobs/ directory exists and is writable
    chmod 755 media/jobs/

Issue: Cleanup not running

Solution:
    Test manually: python manage.py cleanup_expired_jobs
    Check cron logs: grep CRON /var/log/syslog
    Verify cron service is running: systemctl status cron

Issue: PDFs not generating

Solution:
    Check ReportLab/WeasyPrint is installed
    Verify output directory exists: media/jobs/<uuid>/output/
    Check logs for PDF generation errors
"""


# ═══════════════════════════════════════════════════════════════
# VERIFICATION CHECKLIST
# ═══════════════════════════════════════════════════════════════

"""
Before marking refactoring as complete, verify:

[ ] Database migrations applied successfully
[ ] Guest upload page loads at /
[ ] File upload works (drag & drop + click)
[ ] Job creation generates UUID
[ ] Processing shows progress updates
[ ] PDFs generate for all 5 modules
[ ] Download links work
[ ] Confidence score appears in data (0-100%)
[ ] Part A vs Part B counts are tracked
[ ] Auto-cleanup command runs without errors
[ ] Old authenticated features still work (optional)
[ ] Admin panel accessible
[ ] No broken links in navigation
"""


# ═══════════════════════════════════════════════════════════════
# FINAL NOTES
# ═══════════════════════════════════════════════════════════════

"""
This refactoring is BACKWARDS COMPATIBLE with optional authenticated
features. You can keep both:

1. Guest workflow (primary) - NO login required
2. Authenticated workflow (optional) - for users who want accounts

The core functionality works WITHOUT authentication.
Authentication is purely OPTIONAL.

This follows the specification EXACTLY:
- Job-based temporary processing
- No permanent storage of guest uploads
- Auto-cleanup after expiry
- 13-phase workflow
- Confidence score + Part A/B metrics
"""
