"""
Paper models for uploaded question papers.
"""
from django.db import models
from django.conf import settings
from apps.core.models import SoftDeleteModel


class Paper(SoftDeleteModel):
    """Uploaded question paper model."""
    
    class ProcessingStatus(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
    
    subject = models.ForeignKey(
        'subjects.Subject',
        on_delete=models.CASCADE,
        related_name='papers'
    )
    
    # Paper identification
    title = models.CharField(max_length=255)
    year = models.CharField(max_length=50, blank=True)
    exam_type = models.CharField(max_length=100, blank=True)  # e.g., "Mid-term", "Final", "GATE"
    
    # File details
    file = models.FileField(upload_to='papers/')
    file_hash = models.CharField(max_length=64, blank=True)  # SHA-256 hash for dedup
    page_count = models.PositiveIntegerField(default=0)
    
    # Processing status
    status = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.PENDING
    )
    processing_error = models.TextField(blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Detailed progress tracking
    status_detail = models.CharField(max_length=255, blank=True, help_text='Current operation detail')
    questions_extracted = models.PositiveIntegerField(default=0, help_text='Number of questions extracted')
    questions_classified = models.PositiveIntegerField(default=0, help_text='Number of questions classified')
    progress_percent = models.PositiveIntegerField(default=0, help_text='Overall progress percentage')
    
    # Extracted text (cached)
    raw_text = models.TextField(blank=True)
    
    # Metadata
    notes = models.TextField(blank=True)
    
    class Meta:
        verbose_name = 'Paper'
        verbose_name_plural = 'Papers'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['subject', 'status']),
            models.Index(fields=['file_hash']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.year})" if self.year else self.title
    
    def get_question_count(self):
        """Return number of extracted questions."""
        return self.questions.count() if hasattr(self, 'questions') else 0


class PaperPage(models.Model):
    """Individual page from a paper (for OCR/processing)."""
    
    paper = models.ForeignKey(
        Paper,
        on_delete=models.CASCADE,
        related_name='pages'
    )
    page_number = models.PositiveIntegerField()
    text_content = models.TextField(blank=True)
    image = models.ImageField(upload_to='paper_pages/', null=True, blank=True)
    
    class Meta:
        ordering = ['page_number']
        unique_together = ['paper', 'page_number']
    
    def __str__(self):
        return f"Page {self.page_number} of {self.paper.title}"
