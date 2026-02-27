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


# ---------------------------------------------------------------------------
# New production models: ClusterGroup, ClusterMembership, PriorityAssignment
# ---------------------------------------------------------------------------


class ClusterGroup(BaseModel):
    """
    A group of semantically similar questions identified by the clustering
    service.

    CLUSTER TYPES:
    - 'strict_repetition': similarity >= 0.82 AND LLM confirmed SAME with
      confidence >= 80.  These drive study priority assignment.
    - 'concept_similarity': 0.65 <= similarity < 0.82, informational only.
      NEVER used for priority assignment.

    ISOLATION:
    - Each cluster belongs to exactly ONE module and ONE part (A or B).
    - Cross-module and cross-part clustering is forbidden by ClusteringService.
    """

    class ClusterType(models.TextChoices):
        STRICT_REPETITION = 'strict_repetition', 'Strict Repetition (LLM Confirmed)'
        CONCEPT_SIMILARITY = 'concept_similarity', 'Concept Similarity (Informational)'

    subject = models.ForeignKey(
        'subjects.Subject',
        on_delete=models.CASCADE,
        related_name='cluster_groups',
        db_index=True,
    )
    module = models.ForeignKey(
        'subjects.Module',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='cluster_groups',
        db_index=True,
    )

    # Internal label from the clustering algorithm (not necessarily stable)
    cluster_label = models.IntegerField(db_index=True)

    # Cluster type: strict_repetition (priority) or concept_similarity (info only)
    cluster_type = models.CharField(
        max_length=30,
        choices=ClusterType.choices,
        default=ClusterType.STRICT_REPETITION,
        db_index=True,
        help_text=(
            'strict_repetition: verified repeated question (used for priority). '
            'concept_similarity: topically related but not confirmed same (informational).'
        ),
    )

    # Part A or Part B — clusters are NEVER mixed across parts
    part = models.CharField(
        max_length=1,
        default='A',
        db_index=True,
        help_text='Part A or Part B — enforced by ClusteringService isolation rules.',
    )

    # Human-readable representative text (question closest to centroid)
    representative_text = models.TextField(blank=True)

    # Repetition statistics
    frequency = models.PositiveIntegerField(
        default=0,
        help_text='Number of distinct exam years this topic appeared.',
    )
    years_appeared = models.JSONField(default=list, blank=True)
    question_count = models.PositiveIntegerField(default=0)

    class Meta:
        verbose_name = 'Cluster Group'
        verbose_name_plural = 'Cluster Groups'
        ordering = ['-frequency', 'cluster_label']
        indexes = [
            models.Index(fields=['subject', 'module']),
            models.Index(fields=['subject', 'module', 'part']),
            models.Index(fields=['-frequency']),
            models.Index(fields=['cluster_type']),
        ]

    def __str__(self) -> str:
        return (
            f"Cluster {self.cluster_label} [{self.cluster_type}] "
            f"| {self.subject.name} "
            f"| module={self.module_id} part={self.part} "
            f"| freq={self.frequency}"
        )

    @property
    def is_strict_repetition(self) -> bool:
        """True if this cluster represents a confirmed repeated question."""
        return self.cluster_type == self.ClusterType.STRICT_REPETITION


class ClusterMembership(BaseModel):
    """
    Many-to-many link between ClusterGroup and Question.

    Using an explicit through model allows future addition of
    membership-specific fields (e.g., distance to centroid).
    """

    cluster = models.ForeignKey(
        ClusterGroup,
        on_delete=models.CASCADE,
        related_name='members',
        db_index=True,
    )
    question = models.ForeignKey(
        'questions.Question',
        on_delete=models.CASCADE,
        related_name='cluster_memberships',
        db_index=True,
    )

    class Meta:
        verbose_name = 'Cluster Membership'
        verbose_name_plural = 'Cluster Memberships'
        unique_together = [('cluster', 'question')]
        indexes = [
            models.Index(fields=['cluster', 'question']),
        ]

    def __str__(self) -> str:
        return f"Q#{self.question_id} → Cluster {self.cluster.cluster_label}"


class PriorityAssignment(BaseModel):
    """
    Priority tier computed by :class:`~analytics.services.PriorityEngine`
    for a ClusterGroup.

    Stored separately so it can be recomputed without touching cluster data.
    """

    cluster = models.OneToOneField(
        ClusterGroup,
        on_delete=models.CASCADE,
        related_name='priority_assignment',
        db_index=True,
    )

    class Tier(models.IntegerChoices):
        TOP = 1, 'Top Priority (4+ years)'
        HIGH = 2, 'High Priority (3 years)'
        MEDIUM = 3, 'Medium Priority (2 years)'
        LOW = 4, 'Low Priority (1 year)'

    tier = models.IntegerField(choices=Tier.choices, default=Tier.LOW, db_index=True)
    tier_label = models.CharField(max_length=50, blank=True)
    frequency = models.PositiveIntegerField(default=0)

    class Meta:
        verbose_name = 'Priority Assignment'
        verbose_name_plural = 'Priority Assignments'
        ordering = ['tier', '-frequency']
        indexes = [
            models.Index(fields=['tier']),
        ]

    def __str__(self) -> str:
        return f"{self.get_tier_display()} | {self.cluster}"
