"""
Analysis models - JOB-BASED TEMPORARY PROCESSING (NO PERMANENT STORAGE)

This follows the specification:
- Each upload session gets a unique job_id (UUID)
- All data tied to job_id
- Auto-deleted after completion/timeout
- NO user authentication required
"""
import uuid
from django.db import models
from django.utils import timezone
from datetime import timedelta


class AnalysisJob(models.Model):
    """
    Temporary analysis job - represents ONE upload session.
    
    Lifecycle:
    1. User uploads PDFs → job created with UUID
    2. PDFs processed through 13-phase pipeline
    3. Module PDFs generated
    4. User downloads
    5. Auto-cleanup after CLEANUP_TIMEOUT
    """
    
    class Status(models.TextChoices):
        # PHASE 1: Upload
        CREATED = 'created', 'Created'
        UPLOADING = 'uploading', 'Uploading Files'
        
        # PHASE 2: PDF Detection
        DETECTING_PDF_TYPE = 'detecting_pdf_type', 'Detecting PDF Type'
        
        # PHASE 3: Question Segmentation
        EXTRACTING = 'extracting', 'Extracting Questions'
        
        # PHASE 4: Module Mapping
        MAPPING_MODULES = 'mapping_modules', 'Mapping to Modules'
        
        # PHASE 5: Normalization
        NORMALIZING = 'normalizing', 'Normalizing Questions'
        
        # PHASE 6: Embeddings
        EMBEDDING = 'embedding', 'Generating Embeddings'
        
        # PHASE 7: Clustering
        CLUSTERING = 'clustering', 'Clustering Topics'
        
        # PHASE 8-10: Priority Scoring
        SCORING = 'scoring', 'Computing Priority Scores'
        
        # PHASE 11: PDF Generation
        GENERATING_PDFS = 'generating_pdfs', 'Generating Module PDFs'
        
        # Terminal states
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        EXPIRED = 'expired', 'Expired'
    
    # Job identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Status tracking
    status = models.CharField(
        max_length=30,
        choices=Status.choices,
        default=Status.CREATED
    )
    current_phase = models.CharField(max_length=100, blank=True)
    progress = models.PositiveIntegerField(default=0)  # 0-100
    error_message = models.TextField(blank=True)
    
    # Upload metadata (temporary - no user FK!)
    subject_name = models.CharField(max_length=255, default='Unknown Subject', help_text='Subject name from user input')
    pdf_count = models.PositiveIntegerField(default=0)
    total_years = models.PositiveIntegerField(default=0, help_text='Number of distinct years uploaded')
    years_list = models.JSONField(default=list, help_text='List of years for confidence calculation')
    
    # Statistics
    questions_extracted = models.PositiveIntegerField(default=0)
    topics_clustered = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    # Generated PDF paths (temporary)
    output_pdfs = models.JSONField(
        default=dict,
        help_text='Module-wise PDF paths: {"module_1": "/path/", ...}'
    )
    
    class Meta:
        verbose_name = 'Analysis Job'
        verbose_name_plural = 'Analysis Jobs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['expires_at']),
        ]
    
    def __str__(self):
        return f"Job {self.id} - {self.subject_name} ({self.get_status_display()})"
    
    def set_expiry(self, hours=24):
        """Set job expiry time."""
        self.expires_at = timezone.now() + timedelta(hours=hours)
        self.save(update_fields=['expires_at'])
    
    def is_expired(self):
        """Check if job has expired."""
        if not self.expires_at:
            return False
        return timezone.now() > self.expires_at
    
    def mark_completed(self):
        """Mark job as completed and set expiry."""
        self.status = self.Status.COMPLETED
        self.completed_at = timezone.now()
        self.set_expiry(hours=24)  # Auto-delete after 24 hours
        self.save()
    
    def mark_failed(self, error_msg):
        """Mark job as failed."""
        self.status = self.Status.FAILED
        self.error_message = error_msg
        self.completed_at = timezone.now()
        self.set_expiry(hours=1)  # Quick cleanup for failed jobs
        self.save()
