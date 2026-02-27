"""
Priority assignment engine.

Assigns a study-priority tier to each ClusterGroup based on its
repetition frequency across exam years.

CRITICAL: Priority is ONLY assigned to STRICT_REPETITION clusters.
concept_similarity clusters are informational and must NEVER receive
a priority tier assignment — they would inflate study priority
incorrectly.

Tiers are configurable via Django settings:

    PRIORITY_TIER_1_THRESHOLD = 4   # 4+ years → Tier 1 (Top)
    PRIORITY_TIER_2_THRESHOLD = 3   # 3 years  → Tier 2 (High)
    PRIORITY_TIER_3_THRESHOLD = 2   # 2 years  → Tier 3 (Medium)
    # below 2 → Tier 4 (Low)
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


def _get_thresholds() -> tuple[int, int, int]:
    """Read tier thresholds from Django settings with sensible defaults."""
    try:
        from django.conf import settings
        t1 = getattr(settings, "PRIORITY_TIER_1_THRESHOLD", 4)
        t2 = getattr(settings, "PRIORITY_TIER_2_THRESHOLD", 3)
        t3 = getattr(settings, "PRIORITY_TIER_3_THRESHOLD", 2)
        return int(t1), int(t2), int(t3)
    except Exception:
        return 4, 3, 2


def assign_tier(frequency: int) -> int:
    """
    Map a repetition *frequency* to a priority tier integer.

    Args:
        frequency: Number of distinct exam years the topic appeared in.

    Returns:
        Tier integer: 1 (Top) → 2 (High) → 3 (Medium) → 4 (Low).
    """
    t1, t2, t3 = _get_thresholds()
    if frequency >= t1:
        return 1
    if frequency >= t2:
        return 2
    if frequency >= t3:
        return 3
    return 4


TIER_LABELS = {
    1: "Top Priority",
    2: "High Priority",
    3: "Medium Priority",
    4: "Low Priority",
}


class PriorityEngine:
    """
    Assigns priority tiers to ClusterGroup objects and persists
    PriorityAssignment records.

    IMPORTANT: Only processes 'strict_repetition' clusters.
    'concept_similarity' clusters are silently skipped — they must
    never influence study priority.

    Usage::

        engine = PriorityEngine()
        assignments = engine.assign_for_subject(subject)
    """

    def assign_for_subject(self, subject) -> List:
        """
        Compute and persist PriorityAssignment for every
        STRICT_REPETITION ClusterGroup belonging to *subject*.

        Concept_similarity clusters are skipped entirely.
        Existing PriorityAssignment records are replaced (upsert).

        Args:
            subject: ``subjects.Subject`` ORM instance.

        Returns:
            List of upserted PriorityAssignment ORM instances.
        """
        from apps.analytics.models import ClusterGroup, PriorityAssignment

        # ONLY strict_repetition clusters drive study priority
        groups = ClusterGroup.objects.filter(
            subject=subject,
            cluster_type='strict_repetition',
        )

        # Remove stale priority assignments for concept_similarity clusters
        # (in case they were erroneously created by a previous run)
        concept_ids = ClusterGroup.objects.filter(
            subject=subject,
            cluster_type='concept_similarity',
        ).values_list('id', flat=True)
        if concept_ids:
            deleted, _ = PriorityAssignment.objects.filter(
                cluster_id__in=concept_ids
            ).delete()
            if deleted:
                logger.warning(
                    "Removed %d erroneous PriorityAssignment records for "
                    "concept_similarity clusters (subject='%s').",
                    deleted, subject,
                )

        results = []
        for group in groups:
            tier = assign_tier(group.frequency)
            assignment, _ = PriorityAssignment.objects.update_or_create(
                cluster=group,
                defaults={
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "frequency": group.frequency,
                },
            )
            results.append(assignment)

        logger.info(
            "PriorityEngine: assigned tiers for %d strict_repetition cluster "
            "groups (subject='%s'). Concept_similarity clusters: skipped.",
            len(results),
            subject,
        )
        return results

    def assign_for_cluster(self, cluster_group) -> "PriorityAssignment":
        """
        Compute and persist a single PriorityAssignment for *cluster_group*.

        Raises ValueError if the cluster is a concept_similarity cluster
        since those must never get a priority assignment.

        Args:
            cluster_group: ClusterGroup ORM instance.

        Returns:
            Upserted PriorityAssignment instance.
        """
        from apps.analytics.models import PriorityAssignment

        if cluster_group.cluster_type == 'concept_similarity':
            raise ValueError(
                f"PriorityEngine.assign_for_cluster: cluster {cluster_group.pk} "
                f"is a concept_similarity cluster and must not get a priority "
                f"assignment. Only strict_repetition clusters may be prioritised."
            )

        tier = assign_tier(cluster_group.frequency)
        assignment, _ = PriorityAssignment.objects.update_or_create(
            cluster=cluster_group,
            defaults={
                "tier": tier,
                "tier_label": TIER_LABELS[tier],
                "frequency": cluster_group.frequency,
            },
        )
        return assignment

    @staticmethod
    def tier_summary(subject) -> dict:
        """
        Return a summary dict of tier → count for all STRICT_REPETITION
        clusters in *subject*.

        Concept_similarity clusters are excluded from this count.

        Example return value::

            {"Tier 1": 3, "Tier 2": 7, "Tier 3": 12, "Tier 4": 25}

        Args:
            subject: Subject ORM instance.

        Returns:
            Dict mapping tier label → count.
        """
        from apps.analytics.models import PriorityAssignment, ClusterGroup

        summary = {label: 0 for label in TIER_LABELS.values()}
        # Only count strict_repetition-backed assignments
        assignments = PriorityAssignment.objects.filter(
            cluster__subject=subject,
            cluster__cluster_type='strict_repetition',
        )
        for a in assignments:
            label = TIER_LABELS.get(a.tier, "Low Priority")
            summary[label] = summary.get(label, 0) + 1
        return summary

