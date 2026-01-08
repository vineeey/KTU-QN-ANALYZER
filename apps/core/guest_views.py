"""
GUEST UPLOAD VIEWS - NO LOGIN REQUIRED

Implements the exact user flow from specification:
1. User opens website
2. User uploads MULTIPLE PYQ PDFs (same subject)
3. System creates job_id and temporary workspace
4. System analyzes PDFs
5. System generates 5 module-wise PDFs
6. User downloads PDFs
7. System auto-cleans all job data

NO authentication required - pure guest workflow.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import View, TemplateView
from django.http import JsonResponse, FileResponse, Http404
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging

from apps.analysis.models import AnalysisJob
from apps.analysis.job_models import TempPaper
from apps.analysis.pipeline_13phases import (
    Phase1_Upload,
    CompletePipeline
)

logger = logging.getLogger(__name__)


class HomeView(TemplateView):
    """
    Landing page with upload form.
    
    NO login required - anyone can access.
    """
    template_name = 'pages/guest_upload.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'KTU Question Paper Analyzer'
        context['show_instructions'] = True
        return context


class GuestUploadView(View):
    """
    Handle GUEST PDF uploads (no login).
    
    POST /upload/
    - Accepts multiple PDFs
    - Creates job_id (UUID)
    - Starts analysis pipeline
    - Returns job_id for tracking
    """
    
    @method_decorator(csrf_exempt)  # For API-style usage
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """
        Process guest upload.
        
        Form data:
        - subject_name: str (required)
        - pdf_files: MultipleFileField (required)
        """
        # Get subject name
        subject_name = request.POST.get('subject_name', '').strip()
        if not subject_name:
            return JsonResponse({
                'error': 'Subject name is required'
            }, status=400)
        
        # Get uploaded files
        pdf_files = request.FILES.getlist('pdf_files')
        if not pdf_files:
            return JsonResponse({
                'error': 'At least one PDF file is required'
            }, status=400)
        
        if len(pdf_files) > 20:
            return JsonResponse({
                'error': 'Maximum 20 PDFs allowed per upload'
            }, status=400)
        
        try:
            # PHASE 1: Create job and upload PDFs
            job = Phase1_Upload.create_job(subject_name, pdf_files)
            papers = Phase1_Upload.validate_and_save_pdfs(job, pdf_files)
            
            # Extract years from filenames (simple heuristic)
            for paper in papers:
                # Try to extract year from filename (e.g., "2023_Final.pdf")
                import re
                year_match = re.search(r'(20\d{2})', paper.filename)
                if year_match:
                    paper.year = year_match.group(1)
                    paper.exam_name = paper.filename
                    paper.save()
            
            logger.info(f"Created job {job.id} with {len(papers)} papers")
            
            # Start background processing
            # In production: use Celery/Django-Q (tasks.process_job.delay(job.id))
            # In development: run in background thread
            import threading
            def run_pipeline():
                try:
                    CompletePipeline.run_complete_analysis(job.id)
                except Exception as e:
                    logger.exception(f"Pipeline failed for job {job.id}")
                    job.status = AnalysisJob.Status.FAILED
                    job.error_message = str(e)
                    job.save()
            
            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()
            
            # Return job_id for tracking - redirect to RESULTS page (HTML), not status (JSON API)
            return JsonResponse({
                'success': True,
                'job_id': str(job.id),
                'redirect_url': f'/analysis/{job.id}/results/'
            })
            
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.exception("Upload failed")
            return JsonResponse({
                'error': f'Upload failed: {str(e)}'
            }, status=500)


class JobStatusView(View):
    """
    Track job processing status (AJAX polling).
    
    GET /analysis/<job_id>/status/
    - Returns current status
    - NO authentication required (job_id is the auth)
    """
    
    def get(self, request, job_id):
        """Get job status."""
        try:
            job = AnalysisJob.objects.get(id=job_id)
            
            return JsonResponse({
                'job_id': str(job.id),
                'status': job.status,
                'status_display': job.get_status_display(),
                'progress': job.progress,
                'current_phase': job.current_phase,
                'questions_extracted': job.questions_extracted,
                'topics_clustered': job.topics_clustered,
                'error_message': job.error_message,
                'is_completed': job.status == AnalysisJob.Status.COMPLETED,
                'is_failed': job.status == AnalysisJob.Status.FAILED,
                'output_pdfs': job.output_pdfs if job.status == AnalysisJob.Status.COMPLETED else {}
            })
        except AnalysisJob.DoesNotExist:
            return JsonResponse({
                'error': 'Job not found or expired'
            }, status=404)


class JobResultsView(TemplateView):
    """
    Display job results with download links.
    
    GET /analysis/<job_id>/results/
    - Shows module-wise download buttons
    - NO authentication required
    """
    template_name = 'analysis/job_results.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        job_id = kwargs.get('job_id')
        
        try:
            job = AnalysisJob.objects.get(id=job_id)
            
            if job.status != AnalysisJob.Status.COMPLETED:
                context['job'] = job
                context['not_ready'] = True
                return context
            
            context['job'] = job
            context['modules'] = []
            
            # Prepare module download data
            for module_num in range(1, 6):
                pdf_key = f'module_{module_num}'
                if pdf_key in job.output_pdfs:
                    context['modules'].append({
                        'number': module_num,
                        'download_url': f'/analysis/{job.id}/download/{module_num}/',
                        'ready': True
                    })
            
            # Calculate expiry time remaining
            if job.expires_at:
                from django.utils import timezone
                time_remaining = job.expires_at - timezone.now()
                context['hours_remaining'] = int(time_remaining.total_seconds() / 3600)
            
            return context
            
        except AnalysisJob.DoesNotExist:
            context['job_not_found'] = True
            return context


class DownloadModulePDFView(View):
    """
    Download module PDF.
    
    GET /analysis/<job_id>/download/<module_num>/
    - Serves generated PDF
    - NO authentication required
    """
    
    def get(self, request, job_id, module_num):
        """Serve PDF file."""
        try:
            job = AnalysisJob.objects.get(id=job_id)
            
            if job.status != AnalysisJob.Status.COMPLETED:
                raise Http404("Job not completed")
            
            # Get PDF path
            pdf_key = f'module_{module_num}'
            if pdf_key not in job.output_pdfs:
                raise Http404("Module PDF not found")
            
            pdf_path = job.output_pdfs[pdf_key]
            
            # Serve file
            from pathlib import Path
            file_path = Path(pdf_path)
            
            if not file_path.exists():
                raise Http404("PDF file not found")
            
            response = FileResponse(
                open(file_path, 'rb'),
                content_type='application/pdf'
            )
            response['Content-Disposition'] = f'attachment; filename="Module_{module_num}_{job.subject_name}.pdf"'
            
            logger.info(f"Downloaded Module {module_num} for job {job_id}")
            return response
            
        except AnalysisJob.DoesNotExist:
            raise Http404("Job not found or expired")


class StartProcessingView(View):
    """
    Manually start processing (for testing).
    
    POST /analysis/<job_id>/process/
    """
    
    def post(self, request, job_id):
        """Start background processing."""
        try:
            job = AnalysisJob.objects.get(id=job_id)
            
            # Start complete pipeline
            # In production, use: tasks.run_complete_analysis.delay(job.id)
            # For now, run synchronously
            CompletePipeline.run_complete_analysis(job.id)
            
            return JsonResponse({
                'success': True,
                'message': 'Processing started'
            })
            
        except AnalysisJob.DoesNotExist:
            return JsonResponse({
                'error': 'Job not found'
            }, status=404)
        except Exception as e:
            logger.exception("Processing failed")
            return JsonResponse({
                'error': str(e)
            }, status=500)


# ═══════════════════════════════════════════════════════════════
# URL CONFIGURATION (NO LOGIN REQUIRED)
# ═══════════════════════════════════════════════════════════════

from django.urls import path

app_name = 'guest'

urlpatterns = [
    # Home page (upload form)
    path('', HomeView.as_view(), name='home'),
    
    # Upload endpoint
    path('upload/', GuestUploadView.as_view(), name='upload'),
    
    # Job tracking
    path('analysis/<uuid:job_id>/status/', JobStatusView.as_view(), name='job_status'),
    path('analysis/<uuid:job_id>/results/', JobResultsView.as_view(), name='job_results'),
    path('analysis/<uuid:job_id>/process/', StartProcessingView.as_view(), name='start_processing'),
    
    # Download PDFs
    path('analysis/<uuid:job_id>/download/<int:module_num>/', DownloadModulePDFView.as_view(), name='download_module'),
]
