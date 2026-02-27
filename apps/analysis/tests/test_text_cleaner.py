"""
Unit tests for TextCleaner service.

Tests cover:
- normalize_whitespace
- remove_common_artifacts
- rule_based_corrections
- contextual_hook_stub (LLM hook)
- Full pipeline via TextCleaner.clean()
"""
import pytest
from apps.analysis.services.text_cleaner import (
    normalize_whitespace,
    remove_common_artifacts,
    rule_based_corrections,
    contextual_hook_stub,
    TextCleaner,
)


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("Hello   world") == "Hello world"

    def test_strips_lines(self):
        result = normalize_whitespace("  Hello  \n  World  ")
        assert result == "Hello\nWorld"

    def test_preserves_paragraph_breaks(self):
        result = normalize_whitespace("Para 1\n\nPara 2")
        assert "\n\n" in result

    def test_collapses_tabs(self):
        result = normalize_whitespace("Tab\there")
        assert "\t" not in result


class TestRemoveCommonArtifacts:
    def test_removes_standalone_page_number(self):
        text = "Some text\n- 3 -\nMore text"
        result = remove_common_artifacts(text)
        assert "- 3 -" not in result

    def test_removes_page_label(self):
        text = "Question text\nPage 2 of 10\nMore text"
        result = remove_common_artifacts(text)
        assert "Page 2 of 10" not in result

    def test_removes_turn_over(self):
        text = "Question\nTurn over\nAnswer"
        result = remove_common_artifacts(text)
        assert "Turn over" not in result.lower()

    def test_preserves_content(self):
        text = "Define disaster management and explain its importance."
        result = remove_common_artifacts(text)
        assert "Define disaster management" in result


class TestRuleBasedCorrections:
    def test_ligature_fi(self):
        result = rule_based_corrections("deﬁne")
        assert "ﬁ" not in result
        assert "fi" in result

    def test_spaced_question_number(self):
        result = rule_based_corrections("1 . Define hazard")
        assert "1." in result

    def test_broken_hyphen(self):
        result = rule_based_corrections("compu-\nter")
        assert "computer" in result

    def test_let_correction(self):
        result = rule_based_corrections("1et the student explain")
        assert "let" in result


class TestContextualHookStub:
    def test_no_llm_returns_original(self):
        text = "Test text."
        assert contextual_hook_stub(text) == text

    def test_llm_fn_applied(self):
        def mock_llm(t):
            return t.upper()
        result = contextual_hook_stub("hello", llm_fn=mock_llm)
        assert result == "HELLO"

    def test_llm_fn_failure_returns_original(self):
        def bad_llm(t):
            raise RuntimeError("LLM down")
        result = contextual_hook_stub("original", llm_fn=bad_llm)
        assert result == "original"

    def test_llm_fn_empty_returns_original(self):
        result = contextual_hook_stub("original", llm_fn=lambda t: "")
        assert result == "original"


class TestTextCleanerPipeline:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_full_pipeline(self):
        raw = "  1 . Deﬁne disaster   \n- 3 -\nPage 1 of 5\n"
        result = self.cleaner.clean(raw)
        assert "ﬁ" not in result
        assert "Page 1 of 5" not in result
        assert "1." in result or "Define" in result

    def test_idempotent_on_clean_text(self):
        clean = "1. Define disaster management."
        result = self.cleaner.clean(clean)
        assert "Define disaster management" in result
