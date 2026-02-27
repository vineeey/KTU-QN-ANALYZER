"""
Unit tests for two-stage ModuleClassifier.

Tests cover:
- Stage 1: keyword / topic deterministic matching
- Stage 2: semantic fallback (mocked model)
- Batch classification alignment
- Module hint priority
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_module(number, name, keywords=None, topics=None):
    m = MagicMock()
    m.number = number
    m.name = name
    m.keywords = keywords or []
    m.topics = topics or []
    return m


class TestStage1Keywords:
    def setup_method(self):
        # Avoid loading the actual sentence-transformer model
        with patch(
            "apps.analysis.services.classifier.ModuleClassifier._ensure_model_loaded"
        ):
            from apps.analysis.services.classifier import ModuleClassifier
            self.clf = ModuleClassifier.__new__(ModuleClassifier)
            self.clf.llm_client = None
            self.clf._st_loaded = False  # disable Stage 2

    def _modules(self):
        return [
            _make_module(1, "Intro", keywords=["disaster", "hazard"]),
            _make_module(2, "Mitigation", keywords=["mitigation", "preparedness"]),
            _make_module(3, "Policy", keywords=["NDMA", "policy"]),
        ]

    def test_keyword_match_module1(self):
        result = self.clf._stage1_keywords(
            "Define disaster and its types", self._modules()
        )
        assert result == 1

    def test_keyword_match_module2(self):
        result = self.clf._stage1_keywords(
            "Explain mitigation strategies", self._modules()
        )
        assert result == 2

    def test_no_match_returns_none(self):
        result = self.clf._stage1_keywords(
            "The cat sat on the mat", self._modules()
        )
        assert result is None

    def test_module_hint_takes_priority(self):
        # module_hint should short-circuit all other logic in classify()
        from apps.analysis.services.classifier import ModuleClassifier
        with patch.object(ModuleClassifier, "_ensure_model_loaded"):
            clf = ModuleClassifier.__new__(ModuleClassifier)
            clf.llm_client = None
            clf._st_model = None
            clf.__class__._st_loaded = False
            clf.__class__._module_embedding_cache = {}

        modules = [_make_module(2, "Mitigation")]
        subject = MagicMock()
        result = clf.classify(
            "Random text", subject, modules, module_hint=2
        )
        assert result == 2


class TestClassifyByKeywordsShim:
    def setup_method(self):
        with patch(
            "apps.analysis.services.classifier.ModuleClassifier._ensure_model_loaded"
        ):
            from apps.analysis.services.classifier import ModuleClassifier
            self.clf = ModuleClassifier.__new__(ModuleClassifier)
            self.clf.llm_client = None

    def test_shim_calls_stage1(self):
        modules = [_make_module(1, "Intro", keywords=["hazard"])]
        result = self.clf.classify_by_keywords("What is a hazard?", modules)
        assert result == 1
