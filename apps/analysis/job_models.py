"""
Job-scoped temporary models for uploaded PDFs and extracted questions.

These models are TEMPORARY and tied to job_id.
All data auto-deleted when job expires.
"""
from django.db import models
import uuid


class TempPaper(models.Model):
    """
    Temporary uploaded PDF (job-scoped).
    No permanent storage - deleted with job.
    """
    
    class PDFType(models.TextChoices):
        TEXT_BASED = 'text', 'Text-based PDF'
        SCANNED = 'scanned', 'Scanned/Image PDF'
        HYBRID = 'hybrid', 'Hybrid'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Job reference
    job = models.ForeignKey(
        'analysis.AnalysisJob',
        on_delete=models.CASCADE,
        related_name='papers'
    )
    
    # File info
    file = models.FileField(upload_to='jobs/%Y/%m/%d/')  # Will be in /media/jobs/job_id/
    filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(default=0)
    
    # Extracted metadata
    year = models.CharField(max_length=50, blank=True)
    exam_name = models.CharField(max_length=255, blank=True)
    
    # PDF type detection (PHASE 2)
    pdf_type = models.CharField(
        max_length=10,
        choices=PDFType.choices,
        blank=True
    )
    
    # Extracted content (PHASE 2)
    raw_text = models.TextField(blank=True, help_text='Raw extracted text (UNCHANGED)')
    page_count = models.PositiveIntegerField(default=0)
    
    # Processing status
    extracted = models.BooleanField(default=False)
    extraction_error = models.TextField(blank=True)
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['year', 'uploaded_at']
        verbose_name = 'Temporary Paper'
        verbose_name_plural = 'Temporary Papers'
    
    def __str__(self):
        return f"{self.filename} ({self.year})"
    
    def get_workspace_path(self):
        """Get job-specific workspace directory."""
        return f"media/jobs/{self.job_id}/"


class TempQuestion(models.Model):
    """
    Temporary extracted question (job-scoped).
    Represents ONE logical question unit.
    """
    
    class Part(models.TextChoices):
        PART_A = 'A', 'Part A (3 marks)'
        PART_B = 'B', 'Part B (14 marks)'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Source
    paper = models.ForeignKey(
        'TempPaper',
        on_delete=models.CASCADE,
        related_name='questions'
    )
    
    # Question identification (PHASE 3)
    question_number = models.CharField(max_length=20)  # e.g., "1", "11a", "12(i)"
    part = models.CharField(max_length=1, choices=Part.choices)
    marks = models.PositiveIntegerField(default=3)
    
    # Question content (PHASE 3 - RAW, unchanged)
    raw_text = models.TextField(help_text='Original question text with all formatting')
    
    # Normalized text (PHASE 5 - for embeddings)
    normalized_text = models.TextField(
        blank=True,
        help_text='Cleaned text for semantic analysis (no numbers, marks, years)'
    )
    
    # Module assignment (PHASE 4 - rule-based)
    module_number = models.PositiveIntegerField(
        null=True,
        help_text='Module 1-5 based on KTU rules'
    )
    
    # Sub-questions and OR handling
    has_sub_parts = models.BooleanField(default=False)
    sub_parts = models.JSONField(default=list, blank=True)
    is_or_question = models.BooleanField(default=False)
    
    # Images/diagrams
    has_images = models.BooleanField(default=False)
    images = models.JSONField(default=list, blank=True, help_text='Extracted image data')
    
    # Embedding (PHASE 6)
    embedding = models.JSONField(null=True, blank=True, help_text='384-dim vector from sentence-transformer')
    
    # Topic cluster assignment (PHASE 7)
    topic_cluster = models.ForeignKey(
        'TempTopicCluster',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='questions'
    )
    cluster_id = models.IntegerField(null=True, help_text='HDBSCAN cluster ID (-1 = noise)')
    
    # Timestamps
    extracted_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['paper__year', 'question_number']
        indexes = [
            models.Index(fields=['module_number', 'part']),
            models.Index(fields=['cluster_id']),
        ]
    
    def __str__(self):
        return f"Q{self.question_number} ({self.paper.year}) - {self.raw_text[:50]}"
    
    def get_year(self):
        """Get year from parent paper."""
        return self.paper.year


class TempTopicCluster(models.Model):
    """
    Temporary topic cluster (job-scoped).
    Represents one REPEATED TOPIC across multiple years.
    """
    
    class PriorityTier(models.IntegerChoices):
        TIER_1 = 1, 'Very High Priority'
        TIER_2 = 2, 'High Priority'
        TIER_3 = 3, 'Medium Priority'
        TIER_4 = 4, 'Low Priority'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Job reference
    job = models.ForeignKey(
        'analysis.AnalysisJob',
        on_delete=models.CASCADE,
        related_name='topic_clusters'
    )
    
    # Module assignment
    module_number = models.PositiveIntegerField()  # 1-5
    
    # Cluster ID from HDBSCAN
    cluster_id = models.IntegerField(help_text='HDBSCAN cluster ID')
    
    # Topic representation
    topic_label = models.CharField(
        max_length=500,
        help_text='Human-readable topic name (from cluster centroid)'
    )
    representative_question = models.TextField(
        blank=True,
        help_text='Most common question variant'
    )
    
    # PHASE 8: PRIORITY SCORING METRICS
    
    # Frequency (distinct years)
    frequency = models.PositiveIntegerField(
        default=0,
        help_text='Number of DISTINCT YEARS this topic appeared'
    )
    years_appeared = models.JSONField(
        default=list,
        help_text='List of years: ["2021", "2022", ...]'
    )
    
    # Marks statistics
    total_marks = models.PositiveIntegerField(default=0)
    average_marks = models.FloatField(default=0.0)
    
    # Part A vs Part B contribution (PHASE 8.3 - NEW FEATURE)
    part_a_count = models.PositiveIntegerField(
        default=0,
        help_text='How many times appeared in Part A'
    )
    part_b_count = models.PositiveIntegerField(
        default=0,
        help_text='How many times appeared in Part B'
    )
    
    # PHASE 9: CONFIDENCE SCORE (NEW FEATURE)
    confidence_score = models.FloatField(
        default=0.0,
        help_text='(years_appeared / total_years) × 100'
    )
    
    # Priority Score (PHASE 8.5)
    priority_score = models.FloatField(
        default=0.0,
        help_text='(2 × frequency) + average_marks'
    )
    
    # PHASE 10: Priority Tier
    priority_tier = models.IntegerField(
        choices=PriorityTier.choices,
        default=PriorityTier.TIER_4
    )
    
    # Question count
    question_count = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['module_number', 'priority_tier', '-priority_score']
        indexes = [
            models.Index(fields=['module_number', 'priority_tier']),
            models.Index(fields=['-priority_score']),
        ]
        unique_together = ['job', 'module_number', 'cluster_id']
    
    def __str__(self):
        return f"Module {self.module_number} - {self.topic_label} (Tier {self.priority_tier})"
    
    def calculate_metrics(self, total_years_in_job):
        """
        Calculate all metrics (PHASE 8-10).
        
        Args:
            total_years_in_job: Total distinct years uploaded in this job
        """
        questions = self.questions.all()
        
        if not questions.exists():
            return
        
        # Get distinct years
        distinct_years = sorted(set(q.get_year() for q in questions if q.get_year()))
        self.years_appeared = distinct_years
        self.frequency = len(distinct_years)
        
        # Marks statistics
        self.total_marks = sum(q.marks for q in questions)
        self.average_marks = self.total_marks / questions.count() if questions.count() > 0 else 0
        
        # Part A vs Part B contribution (NEW FEATURE)
        self.part_a_count = questions.filter(part='A').count()
        self.part_b_count = questions.filter(part='B').count()
        
        # Confidence score (NEW FEATURE)
        if total_years_in_job > 0:
            self.confidence_score = (self.frequency / total_years_in_job) * 100
        else:
            self.confidence_score = 0.0
        
        # Priority score
        self.priority_score = (2 * self.frequency) + self.average_marks
        
        # Priority tier assignment
        if self.frequency >= 4:
            self.priority_tier = self.PriorityTier.TIER_1
        elif self.frequency >= 3:
            self.priority_tier = self.PriorityTier.TIER_2
        elif self.frequency >= 2:
            self.priority_tier = self.PriorityTier.TIER_3
        else:
            self.priority_tier = self.PriorityTier.TIER_4
        
        self.question_count = questions.count()
        self.save()
    
    def get_tier_display_full(self):
        """Get full tier description for PDF output."""
        tier_labels = {
            1: "🔥 VERY HIGH PRIORITY (Tier 1)",
            2: "⚡ HIGH PRIORITY (Tier 2)",
            3: "📌 MEDIUM PRIORITY (Tier 3)",
            4: "📝 LOW PRIORITY (Tier 4)",
        }
        return tier_labels.get(self.priority_tier, "Unknown")
    
    def get_part_distribution_text(self):
        """Get Part A vs Part B distribution for PDF."""
        return f"Part A: {self.part_a_count} times | Part B: {self.part_b_count} times"
