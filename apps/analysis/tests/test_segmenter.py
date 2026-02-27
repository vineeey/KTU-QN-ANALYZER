"""
Unit tests for Segmenter service.

Tests cover:
- Part A / Part B detection
- Question number parsing
- Sub-question detection
- Multi-line continuation
- Module hint extraction
- DTO validity
"""
import pytest
from apps.analysis.services.segmenter import (
    Segmenter,
    QuestionDTO,
    _extract_marks,
    _strip_marks,
)


class TestExtractMarks:
    def test_parenthesis_marks(self):
        assert _extract_marks("Explain the concept. (5 marks)") == 5

    def test_bracket_marks(self):
        assert _extract_marks("Define disaster. [10]") == 10

    def test_no_marks(self):
        assert _extract_marks("What is a hazard?") is None

    def test_marks_colon(self):
        assert _extract_marks("Answer the following. Marks: 3") == 3


class TestStripMarks:
    def test_removes_parenthesis(self):
        result = _strip_marks("Explain floods. (5 marks)")
        assert "(5 marks)" not in result
        assert "Explain floods" in result

    def test_removes_bracket(self):
        result = _strip_marks("Define risk. [10]")
        assert "[10]" not in result


class TestQuestionDTO:
    def test_valid_dto(self):
        dto = QuestionDTO(
            question_number="1",
            text="What is a disaster?",
            part="A",
            marks=2,
        )
        assert dto.is_valid()

    def test_invalid_dto_empty_text(self):
        dto = QuestionDTO(question_number="1", text="", part="A")
        assert not dto.is_valid()

    def test_invalid_dto_short_text(self):
        dto = QuestionDTO(question_number="1", text="Hi", part="A")
        assert not dto.is_valid()

    def test_to_dict(self):
        dto = QuestionDTO(
            question_number="3",
            text="Describe the disaster management cycle.",
            part="B",
            marks=10,
            module_hint=2,
        )
        d = dto.to_dict()
        assert d["question_number"] == "3"
        assert d["marks"] == 10
        assert d["module_hint"] == 2


class TestSegmenter:
    def setup_method(self):
        self.seg = Segmenter()

    def test_basic_part_a_extraction(self):
        text = """
PART A
1. Define disaster.
2. What is hazard?
"""
        dtos = self.seg.parse(text)
        assert len(dtos) == 2
        assert all(d.part == "A" for d in dtos)

    def test_part_switch(self):
        text = """
PART A
1. Define disaster.
PART B
11. Explain the types of natural disasters. (10 marks)
"""
        dtos = self.seg.parse(text)
        assert any(d.part == "A" for d in dtos)
        assert any(d.part == "B" for d in dtos)

    def test_marks_extracted(self):
        text = "PART B\n11. Explain disaster management cycle. (10 marks)"
        dtos = self.seg.parse(text)
        assert len(dtos) == 1
        assert dtos[0].marks == 10

    def test_sub_questions(self):
        text = """
PART B
11. Answer the following:
    (a) Define hazard.
    (b) Explain vulnerability.
"""
        dtos = self.seg.parse(text)
        assert len(dtos) == 1
        assert len(dtos[0].sub_questions) == 2

    def test_module_hint(self):
        text = """
Module 3
PART B
11. Discuss NDMA structure.
"""
        dtos = self.seg.parse(text)
        assert len(dtos) == 1
        assert dtos[0].module_hint == 3

    def test_multiline_continuation(self):
        text = """
PART B
11. Explain the phases of disaster management
with examples from recent events.
"""
        dtos = self.seg.parse(text)
        assert len(dtos) == 1
        assert "phases of disaster management" in dtos[0].text
        assert "recent events" in dtos[0].text

    def test_empty_text(self):
        dtos = self.seg.parse("")
        assert dtos == []

    def test_no_questions(self):
        dtos = self.seg.parse("Random text without question numbers.")
        assert dtos == []
