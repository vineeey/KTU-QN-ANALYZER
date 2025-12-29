"""
Analytics models for topic clustering and repetition analysis.
"""
from django.db import models
from apps.core.models import BaseModel


class TopicCluster(BaseModel):
    """
    Represents a cluster of similar questions grouped as a 'topic'.
    Used for repetition analysis and priority assignment.
    """
    
    class PriorityTier(models.IntegerChoices):
        TIER_1 = 1, 'Top Priority (4+ exams)'
        TIER_2 = 2, 'High Priority (3 exams)'
        TIER_3 = 3, 'Medium Priority (2 exams)'
        TIER_4 = 4, 'Low Priority (1 exam)'
    
    subject = models.ForeignKey(
        'subjects.Subject',
        on_delete=models.CASCADE,
        related_name='topic_clusters'
    )
    
    module = models.ForeignKey(
        'subjects.Module',
        on_delete=models.CASCADE,
        related_name='topic_clusters',
        null=True,
        blank=True
    )
    
    # Human-readable topic label (e.g., "Layers of atmosphere")
    topic_name = models.CharField(max_length=500)
    
    # Normalized key for similarity matching
    normalized_key = models.CharField(max_length=500, db_index=True)
    
    # Representative question text (typically the most common variant)
    representative_text = models.TextField(blank=True)
    
    # Questions belonging to this cluster
    # Stored as references via Question.topic_cluster FK
    
    # Repetition statistics
    frequency_count = models.PositiveIntegerField(default=0, help_text='Number of exams where this topic appears')
    years_appeared = models.JSONField(default=list, blank=True, help_text='List of years/exam names')
    total_marks = models.PositiveIntegerField(default=0, help_text='Total marks across all occurrences')
    question_count = models.PositiveIntegerField(default=0, help_text='Total number of questions in this cluster')
    
    # Priority tier (calculated from frequency_count)
    priority_tier = models.IntegerField(
        choices=PriorityTier.choices,
        default=PriorityTier.TIER_4,
        help_text='1=Top, 2=High, 3=Medium, 4=Low'
    )
    
    # Cluster identifier from clustering algorithm
    cluster_id = models.CharField(max_length=100, blank=True, help_text='Cluster ID from algorithm')
    
    # Part distribution (how many times in Part A vs Part B)
    part_a_count = models.PositiveIntegerField(default=0)
    part_b_count = models.PositiveIntegerField(default=0)
    
    class Meta:
        verbose_name = 'Topic Cluster'
        verbose_name_plural = 'Topic Clusters'
        ordering = ['-frequency_count', 'topic_name']
        indexes = [
            models.Index(fields=['subject', 'module']),
            models.Index(fields=['priority_tier']),
            models.Index(fields=['-frequency_count']),
        ]
    
    def __str__(self):
        return f"{self.topic_name} ({self.get_priority_tier_display()})"
    
    def calculate_priority_tier(self, tier_1_threshold=4, tier_2_threshold=3, tier_3_threshold=2):
        """Calculate and set priority tier based on frequency count."""
        if self.frequency_count >= tier_1_threshold:
            self.priority_tier = self.PriorityTier.TIER_1
        elif self.frequency_count >= tier_2_threshold:
            self.priority_tier = self.PriorityTier.TIER_2
        elif self.frequency_count >= tier_3_threshold:
            self.priority_tier = self.PriorityTier.TIER_3
        else:
            self.priority_tier = self.PriorityTier.TIER_4
    
    def get_questions(self):
        """Return all questions in this cluster."""
        return self.questions.all()
    
    def get_tier_label(self):
        """Get a human-readable tier label."""
        tier_map = {
            self.PriorityTier.TIER_1: 'Top Priority',
            self.PriorityTier.TIER_2: 'High Priority',
            self.PriorityTier.TIER_3: 'Medium Priority',
            self.PriorityTier.TIER_4: 'Low Priority',
        }
        return tier_map.get(self.priority_tier, 'Unknown')
