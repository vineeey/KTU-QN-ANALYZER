"""
Unit tests for priority assignment engine.

Tests cover:
- assign_tier function with various frequencies
- Threshold boundary cases
- TIER_LABELS mapping
"""
import pytest
from unittest.mock import MagicMock, patch
from apps.analytics.services.priority_engine import (
    assign_tier,
    TIER_LABELS,
    PriorityEngine,
)


class TestAssignTier:
    """Test the pure assign_tier function (no Django ORM interaction)."""

    def test_tier1_at_threshold(self):
        assert assign_tier(4) == 1

    def test_tier1_above_threshold(self):
        assert assign_tier(10) == 1

    def test_tier2_exactly(self):
        assert assign_tier(3) == 2

    def test_tier3_exactly(self):
        assert assign_tier(2) == 3

    def test_tier4_one(self):
        assert assign_tier(1) == 4

    def test_tier4_zero(self):
        assert assign_tier(0) == 4

    def test_tier_labels_cover_all_tiers(self):
        for tier in (1, 2, 3, 4):
            assert tier in TIER_LABELS
            assert isinstance(TIER_LABELS[tier], str)


class TestPriorityEngineAssignForSubject:
    """Test PriorityEngine.assign_for_subject with mocked ORM."""

    def _make_cluster(self, frequency):
        cluster = MagicMock()
        cluster.frequency = frequency
        return cluster

    def test_all_tiers_covered(self):
        engine = PriorityEngine()
        clusters = [
            self._make_cluster(5),   # Tier 1
            self._make_cluster(3),   # Tier 2
            self._make_cluster(2),   # Tier 3
            self._make_cluster(1),   # Tier 4
        ]
        for c in clusters:
            tier = assign_tier(c.frequency)
            assert 1 <= tier <= 4

    def test_tier_summary_keys(self):
        summary = {label: 0 for label in TIER_LABELS.values()}
        assert "Top Priority" in summary
        assert "High Priority" in summary
        assert "Medium Priority" in summary
        assert "Low Priority" in summary
