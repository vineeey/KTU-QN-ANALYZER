"""
Question models with full analysis fields.
"""
from django.db import models
from apps.core.models import BaseModel


class Question(BaseModel):
    """Extracted question with all analysis fields."""
    
    class DifficultyLevel(models.TextChoices):
        EASY = 'easy', 'Easy'
        MEDIUM = 'medium', 'Medium'
        HARD = 'hard', 'Hard'
    
    class BloomLevel(models.TextChoices):
        REMEMBER = 'remember', 'Remember'
        UNDERSTAND = 'understand', 'Understand'
        APPLY = 'apply', 'Apply'
        ANALYZE = 'analyze', 'Analyze'
        EVALUATE = 'evaluate', 'Evaluate'
        CREATE = 'create', 'Create'
    
    class QuestionType(models.TextChoices):
        DEFINITION = 'definition', 'Definition'
        DERIVATION = 'derivation', 'Derivation'
        NUMERICAL = 'numerical', 'Numerical Problem'
        THEORY = 'theory', 'Theoretical'
        DIAGRAM = 'diagram', 'Diagram-based'
        COMPARISON = 'comparison', 'Comparison'
        SHORT_ANSWER = 'short_answer', 'Short Answer'
        LONG_ANSWER = 'long_answer', 'Long Answer'
    
    # Source
    paper = models.ForeignKey(
        'papers.Paper',
        on_delete=models.CASCADE,
        related_name='questions'
    )
    
    # Question content
    question_number = models.CharField(max_length=20, blank=True)
    text = models.TextField()
    sub_questions = models.JSONField(default=list, blank=True)
    marks = models.PositiveIntegerField(null=True, blank=True)
    
    # OCR tracking fields
    raw_ocr_text = models.TextField(
        blank=True,
        help_text='Original OCR output before LLM cleaning'
    )
    cleaned_text = models.TextField(
        blank=True,
        help_text='LLM-cleaned text (if OCR cleaning was used)'
    )
    ocr_cleaned_by = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ('none', 'Not Cleaned'),
            ('gemini', 'Gemini 1.5 Flash'),
            ('ollama', 'Ollama'),
        ],
        default='none',
        help_text='Which LLM was used for OCR cleaning'
    )
    
    # Images and diagrams (extracted coordinates and data)
    images = models.JSONField(
        default=list, 
        blank=True,
        help_text='List of extracted images with bbox coordinates and base64 data'
    )
    
    # Classification
    module = models.ForeignKey(
        'subjects.Module',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='questions'
    )
    topics = models.JSONField(default=list, blank=True)
    keywords = models.JSONField(default=list, blank=True)
    
    # Question type classification (via LLM)
    question_type = models.CharField(
        max_length=20,
        choices=QuestionType.choices,
        blank=True,
        help_text='Type of question classified by LLM'
    )
    
    # Analysis results
    difficulty = models.CharField(
        max_length=10,
        choices=DifficultyLevel.choices,
        blank=True
    )
    bloom_level = models.CharField(
        max_length=15,
        choices=BloomLevel.choices,
        blank=True
    )
    
    # Embedding for similarity search
    embedding = models.JSONField(null=True, blank=True)  # Store as list of floats
    
    # Duplicate detection
    is_duplicate = models.BooleanField(default=False)
    duplicate_of = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='duplicates'
    )
    similarity_score = models.FloatField(null=True, blank=True)
    
    # Similarity detection tracking
    similarity_method = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ('embedding', 'Embedding Only'),
            ('llm', 'LLM Only'),
            ('hybrid', 'Hybrid (Embedding + LLM)'),
        ],
        help_text='Method used for similarity detection'
    )
    similarity_reason = models.TextField(
        blank=True,
        help_text='LLM explanation for similarity decision (for edge cases)'
    )
    
    # Repetition tracking
    repetition_count = models.PositiveIntegerField(
        default=0,
        help_text='Number of times this question appeared across papers'
    )
    years_appeared = models.JSONField(
        default=list,
        blank=True,
        help_text='List of years when this question appeared'
    )
    
    # Importance and frequency
    importance_score = models.FloatField(
        default=0.0,
        help_text='Calculated importance score based on frequency and recency'
    )
    frequency_score = models.FloatField(
        default=0.0,
        help_text='Frequency score across years'
    )
    
    # Topic cluster (for repetition analysis)
    topic_cluster = models.ForeignKey(
        'analytics.TopicCluster',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='questions'
    )
    
    # Part (A or B) - extracted from question paper
    part = models.CharField(max_length=1, blank=True, help_text='Part A or Part B')
    
    # Manual override flags
    module_manually_set = models.BooleanField(default=False)
    difficulty_manually_set = models.BooleanField(default=False)
    
    class Meta:
        verbose_name = 'Question'
        verbose_name_plural = 'Questions'
        ordering = ['question_number']
        indexes = [
            models.Index(fields=['paper', 'module']),
            models.Index(fields=['is_duplicate']),
            models.Index(fields=['difficulty', 'bloom_level']),
        ]
    
    def __str__(self):
        return f"Q{self.question_number}: {self.text[:50]}..."
    
    def clean(self):
        """Validate question data."""
        from django.core.exceptions import ValidationError
        super().clean()
        
        # Validate marks are positive
        if self.marks is not None and self.marks <= 0:
            raise ValidationError({'marks': 'Marks must be positive'})
        
        # Validate question text is not empty
        if not self.text or len(self.text.strip()) < 5:
            raise ValidationError({'text': 'Question text is too short'})
        
        # Validate duplicate relationship
        if self.is_duplicate and not self.duplicate_of:
            raise ValidationError({'duplicate_of': 'Duplicate questions must reference original'})
    
    def save(self, *args, **kwargs):
        """Override save to perform validation."""
        # Only perform full_clean if not in a bulk operation
        if not kwargs.pop('skip_validation', False):
            self.full_clean()
        super().save(*args, **kwargs)
    
    def get_similar_questions(self, threshold=0.8):
        """Find similar questions based on embedding similarity."""
        if not self.embedding:
            return Question.objects.none()
        # This would be implemented using the embedding service
        return Question.objects.none()
