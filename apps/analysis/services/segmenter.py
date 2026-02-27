"""
Strict stateful question segmenter for KTU exam paper OCR text.

DESIGN RULES (mandatory):
- Module assignment comes ONLY from explicit module headings ("Module 1",
  "Unit III") found in the paper text.  The segmenter sets
  ``module_hint_is_explicit = True`` in that case.
  → Tasks layer must NEVER override an explicit module hint with the
    semantic classifier.
- Sub-questions (a), (b), (i), (ii) stay grouped under the same parent
  question.  They are stored in ``sub_questions`` AND concatenated into
  ``full_question_text`` so the embedding captures the full context.
- Part A and Part B are tracked separately via the ``part`` field.
- Question numbering ("1.", "Q1)", "a)") is removed from stored text.
- Marks annotations are already removed by TextCleaner before this runs;
  the segmenter does NOT strip marks itself (belt-and-suspenders only).
- The segmenter never semantically reclassifies questions.

Input formats handled:
1. KTU module-summary PDFs: bullet-style with year annotations.
2. Standard numbered exam papers: "1. text", "Q1) text".

Output: List[QuestionDTO] where every DTO has:
    year              – exam session (e.g. "December 2022")
    module            – explicit module number from heading, or None
    module_hint_is_explicit – True iff module came from a paper heading
    part              – "A" or "B"
    full_question_text – complete clean text, no numbering, no marks
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Object
# ---------------------------------------------------------------------------


@dataclass
class QuestionDTO:
    """
    Structured representation of a single extracted question.

    Produced by :class:`Segmenter`.  Django-free so the module stays
    independently testable.

    Fields
    ------
    question_number
        Internal counter string (used for DB uniqueness, never shown).
    text
        Core question text — no numbering, no marks annotation.
    part
        "A" (short, 2–3 marks) or "B" (long, 7–14 marks).
    marks
        Numeric marks value extracted from the annotation, or a default.
    sub_questions
        List of sub-question texts (a), (b), etc.
    module_hint
        Explicit module number from the paper's own "Module N" heading.
        ``None`` if no heading was found for this question.
    module_hint_is_explicit
        True iff ``module_hint`` was set from an actual heading in the
        paper (NOT from semantic/keyword inference).  When True, the
        pipeline layer MUST NOT override it with the classifier.
    year_context
        Exam session string, e.g. "December 2022".
    qn_group
        Part-B question group number ("11", "12", …).
    full_question_text
        Complete text ready for embedding: ``text`` with all sub-question
        texts concatenated.  This is what gets embedded — not ``text``
        alone.
    """

    question_number: str
    text: str
    part: str                              # "A" or "B"
    marks: Optional[int] = None
    sub_questions: List[str] = field(default_factory=list)
    module_hint: Optional[int] = None
    module_hint_is_explicit: bool = False  # True = set from paper heading
    year_context: Optional[str] = None
    qn_group: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)

    # Computed on first access or after sub-question appends
    _full_text_cache: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def full_question_text(self) -> str:
        """
        Return the complete question text including all sub-questions,
        joined with a space.  This is the canonical text for embedding.
        """
        parts = [self.text.strip()]
        for sq in self.sub_questions:
            sq = sq.strip()
            if sq:
                parts.append(sq)
        return " ".join(p for p in parts if p)

    def is_valid(self) -> bool:
        return bool(self.text and self.text.strip() and len(self.text.strip()) > 5)

    def to_dict(self) -> dict:
        return {
            "question_number": self.question_number,
            "text": self.text.strip(),
            "full_question_text": self.full_question_text,
            "part": self.part,
            "marks": self.marks,
            "sub_questions": self.sub_questions,
            "module_hint": self.module_hint,
            "module_hint_is_explicit": self.module_hint_is_explicit,
            "year_context": self.year_context,
            "qn_group": self.qn_group,
        }


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_RE_PART_A = re.compile(r"(?i)\bpart\s*[-:]?\s*a\b")
_RE_PART_B = re.compile(r"(?i)\bpart\s*[-:]?\s*b\b")

# Module heading: "Module 1", "Module III", "Unit 2", "Unit IV"
_RE_MODULE = re.compile(r"(?i)\b(module|unit)\s*[-:]?\s*([1-9ivxlcdm]+)\b")

# Standard numbered question:  "1.  text",  "Q2)  text",  "Q. 3  text"
_RE_QUESTION_START = re.compile(
    r"^(?:Q\.?\s*)?(\d{1,2})\s*[.)]\s+(.+)", re.IGNORECASE
)

# Sub-question:  "(a) text",  "a) text",  "i. text",  "(i) text"
_RE_SUB_QUESTION = re.compile(
    r"^\s*\(?([a-z]|i{1,3}v?|vi{0,3}|ix|x)\)\.?\s+(.+)", re.IGNORECASE
)

# Bullet question:  "• text",  "● text"
# Also "¢" which PaddleOCR sometimes outputs for printed bullets.
_RE_BULLET = re.compile(r"^[•●¢]\s+(.{5,})")
# OCR dash/em-dash/asterisk at line start with 10+ chars of content.
_RE_DASH_BULLET = re.compile(r"^(?:\*|–|—)\s+(.{10,})")

# Plain text question ending with marks annotation (already removed by
# TextCleaner, but kept as belt-and-suspenders).
_RE_MARKS_LINE = re.compile(
    r"^(.{8,}?)\s+[—–\-]\s*"
    r"\(\s*(?:[A-Za-z]{3,9}\.?\s+20\d{2}\s*,\s*)?"
    r"(\d+)\s*marks?\s*\)\s*$",
    re.IGNORECASE,
)

# Year header:  "December 2021",  "N December 2021",  "◆ June 2024"
_RE_YEAR_HEADER = re.compile(
    r"(?:^[N◆✦►●\-]?\s*)?"
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+(20\d{2})",
    re.IGNORECASE,
)

# Part-B question group:  "Qn 11",  "Qn.12"
_RE_QN_MARKER = re.compile(r"^Qn\.?\s*(\d{1,2})\s*$", re.IGNORECASE)

# Marks annotation at end of bullet (for extraction, not removal)
_RE_MARKS_AT_END = re.compile(
    r"[—–\-]?\s*\("
    r"(?:[A-Za-z]{3,9}\.?\s+\d{4},\s*)?"
    r"(\d+)\s*(?:[×x\*]\s*\d+\s*=\s*\d+\s*)?marks?"
    r"\)\s*$",
    re.IGNORECASE,
)

# Generic marks anywhere
_RE_MARKS_GENERIC = re.compile(
    r"(?:\((\d+)\s*[×x\*]?\s*\d*\s*=?\s*\d*\s*marks?\)|\[(\d+)\]|marks\s*:\s*(\d+))",
    re.IGNORECASE,
)

# Stop-parse: analysis/priority section starts
_RE_STOP_PARSE = re.compile(
    r"(?i)(repeated\s+questions|prioritized\s+list|final\s+prioritized|"
    r"tier\s+[1-4]|top\s+priority|high\s+priority|medium\s+priority|"
    r"low\s+priority|appears\s+in:|appears\s+only\s+once|study\s+order)",
)

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def _roman_to_int(s: str) -> Optional[int]:
    s = s.lower().strip()
    return int(s) if s.isdigit() else _ROMAN.get(s)


def _extract_marks(line: str) -> Optional[int]:
    """Return the first marks value found in a line, or None."""
    m = _RE_MARKS_AT_END.search(line)
    if m:
        return int(m.group(1))
    m = _RE_MARKS_GENERIC.search(line)
    if m:
        for g in m.groups():
            if g and str(g).isdigit():
                return int(g)
    return None


def _strip_marks(line: str) -> str:
    """Remove all marks annotations from a line."""
    stripped = _RE_MARKS_AT_END.sub("", line).strip()
    stripped = _RE_MARKS_GENERIC.sub("", stripped).strip()
    stripped = re.sub(r"[—–\-,]\s*$", "", stripped).strip()
    return stripped


def _strip_question_number(text: str) -> str:
    """
    Remove leading question number/letter prefix from text.

    Handles:
    - "1. text" → "text"
    - "Q1) text" → "text"
    - "a) text" → "text"
    - "i. text" → "text"
    """
    # Standard numbered: "1. " or "Q1) "
    text = re.sub(r"^(?:Q\.?\s*)?\d{1,2}\s*[.)]\s*", "", text, flags=re.IGNORECASE)
    # Sub-question: "(a) " or "a) " or "i. "
    text = re.sub(r"^\(?([a-z]|i{1,3}v?|vi{0,3}|ix|x)\)\.?\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_year_from_line(line: str) -> Optional[str]:
    """Extract exam session label from bullet text, e.g. 'Dec 2022' → 'December 2022'."""
    month_map = {
        "jan": "January", "feb": "February", "mar": "March", "apr": "April",
        "may": "May", "jun": "June", "jul": "July", "aug": "August",
        "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    }
    m = re.search(r"\(([A-Za-z]{3,9})\.?\s+(20\d{2}),", line, re.IGNORECASE)
    if m:
        mon = m.group(1).lower()[:3]
        yr = m.group(2)
        full_month = month_map.get(mon)
        return f"{full_month} {yr}" if full_month else f"{m.group(1)} {yr}"
    return None


# ---------------------------------------------------------------------------
# Parser state
# ---------------------------------------------------------------------------


class _ParserState:
    def __init__(self):
        self.current_part: str = "A"
        self.current_module: Optional[int] = None
        self.module_is_explicit: bool = False  # True when set from paper heading
        self.current_year: Optional[str] = None
        self.current_qn_group: Optional[str] = None
        self.bullet_counter: int = 0
        self.current_q: Optional[QuestionDTO] = None
        self.finished: List[QuestionDTO] = []
        self.stop: bool = False

    def flush(self):
        if self.current_q and self.current_q.is_valid():
            self.finished.append(self.current_q)
        self.current_q = None


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class Segmenter:
    """
    Stateful parser that converts cleaned OCR text into :class:`QuestionDTO`
    objects.

    Key guarantees:
    - Module is set ONLY from explicit paper headings (module_hint_is_explicit=True).
    - Sub-questions remain attached to their parent question.
    - Part A and Part B are tracked separately.
    - Stored question text never contains question numbering or marks.
    - No questions are merged across Part/Module boundaries.

    Usage::

        seg = Segmenter()
        dtos = seg.parse(cleaned_text)
    """

    # Patterns for pre-pass
    _RE_LINE_ENDS_WITH_MARKS = re.compile(
        r"\(\s*(?:[A-Za-z]{3,9}\.?\s+20\d{2}\s*,\s*)?\d+\s*marks?\s*\)\s*$",
        re.IGNORECASE,
    )
    _RE_ENDS_WITH_DASH = re.compile(r"[-—–]\s*$")
    _RE_IS_ANNOTATION_TAIL = re.compile(
        r"^\s*(?:\(?\s*[A-Za-z]{3,9}\.?\s+20\d{2}\s*,\s*)?\d*\s*marks?\s*\)\s*$",
        re.IGNORECASE,
    )
    _RE_IS_STRUCTURAL = re.compile(
        r"^(?:"
        r"Part\s*[AB]|"
        r"Module\s+\d|"
        r"Unit\s+[IVX\d]+|"
        r"Qn\.?\s*\d|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+20\d{2}"
        r")",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> List[QuestionDTO]:
        """
        Parse cleaned OCR text into QuestionDTO objects.

        Two pre-passes reassemble questions that PaddleOCR split across
        multiple lines, then the main state machine runs.
        """
        lines = text.splitlines()

        # ── Pass 1: stitch annotation tail split onto previous line ──────
        stitched: List[str] = []
        for ln in lines:
            s = ln.strip()
            if stitched:
                prev = stitched[-1].rstrip()
                if (
                    self._RE_IS_ANNOTATION_TAIL.match(s)
                    or (s.startswith("(") and self._RE_LINE_ENDS_WITH_MARKS.search(s))
                ):
                    stitched[-1] = prev + " " + s
                    continue
            stitched.append(ln)

        # ── Pass 2: join multi-line questions ────────────────────────────
        merged: List[str] = []
        _buf: List[str] = []

        def _flush_buf():
            if not _buf:
                return
            last = _buf[-1].strip()
            if self._RE_LINE_ENDS_WITH_MARKS.search(last):
                merged.append(" ".join(p.strip() for p in _buf))
            else:
                merged.extend(_buf)
            _buf.clear()

        for ln in stitched:
            s = ln.strip()
            if not s:
                _flush_buf()
                merged.append("")
                continue
            if self._RE_IS_STRUCTURAL.match(s) or _RE_STOP_PARSE.search(s):
                _flush_buf()
                merged.append(ln)
                continue
            if _RE_BULLET.match(s) or _RE_DASH_BULLET.match(s) or _RE_QUESTION_START.match(s):
                _flush_buf()
                merged.append(ln)
                continue
            _buf.append(s)
            if self._RE_LINE_ENDS_WITH_MARKS.search(s):
                _flush_buf()

        _flush_buf()
        text = "\n".join(merged)

        # ── Main state machine ────────────────────────────────────────────
        state = _ParserState()
        for line in text.splitlines():
            if state.stop:
                break
            self._process_line(line, state)
        state.flush()

        logger.info(
            "Segmentation complete: %d questions extracted "
            "(explicit module headings: %d)",
            len(state.finished),
            sum(1 for d in state.finished if d.module_hint_is_explicit),
        )
        return state.finished

    def _process_line(self, raw_line: str, state: _ParserState) -> None:
        stripped = raw_line.strip()
        if not stripped:
            return

        # ── Stop trigger ────────────────────────────────────────────────
        if _RE_STOP_PARSE.search(stripped):
            state.stop = True
            state.flush()
            return

        # ── Part headers ─────────────────────────────────────────────────
        if _RE_PART_A.match(stripped):
            state.flush()
            state.current_part = "A"
            state.current_qn_group = None
            return
        if _RE_PART_B.match(stripped):
            state.flush()
            state.current_part = "B"
            state.current_qn_group = None
            return

        # ── Explicit Module heading ───────────────────────────────────────
        # CRITICAL: module is set ONLY from this explicit heading.
        # The semantic classifier in tasks.py must NOT override when
        # module_hint_is_explicit is True.
        mod_m = _RE_MODULE.match(stripped)
        if mod_m:
            module_num = _roman_to_int(mod_m.group(2))
            if module_num:
                state.current_module = module_num
                state.module_is_explicit = True
                logger.debug(
                    "Explicit module heading found: Module %d", module_num
                )
            # Don't return — a module heading line sometimes contains a
            # year or can precede an inline question; let parsing continue.

        # ── Year header ──────────────────────────────────────────────────
        yr_m = _RE_YEAR_HEADER.search(stripped)
        if yr_m:
            state.current_year = f"{yr_m.group(1).capitalize()} {yr_m.group(2)}"
            remaining = _RE_YEAR_HEADER.sub("", stripped).strip()
            remaining = re.sub(r"^[N◆✦►●\-\s]+", "", remaining).strip()
            if not remaining:
                return
            # If there's remaining text after the year, treat it as the
            # next line to process (fall through)
            stripped = remaining

        # ── Part-B question group marker: "Qn 11" ───────────────────────
        qn_m = _RE_QN_MARKER.match(stripped)
        if qn_m:
            state.flush()
            state.current_qn_group = qn_m.group(1)
            return

        # ── Bullet question (•/● or PaddleOCR –/*/—) ────────────────────
        bul_m = _RE_BULLET.match(stripped) or _RE_DASH_BULLET.match(stripped)
        if bul_m:
            state.flush()
            state.bullet_counter += 1
            raw_content = bul_m.group(1).strip()
            year = _extract_year_from_line(raw_content) or state.current_year
            marks = _extract_marks(raw_content)
            # Strip marks and year annotation from stored text
            clean_text = _strip_marks(raw_content).strip()
            if marks is None:
                marks = 3 if state.current_part == "A" else 14

            if not clean_text or len(clean_text.strip()) < 8:
                return

            if state.current_part == "B" and state.current_qn_group:
                q_num = f"{state.current_qn_group}b{state.bullet_counter}"
            else:
                q_num = str(state.bullet_counter)

            state.current_q = QuestionDTO(
                question_number=q_num,
                text=clean_text,
                part=state.current_part,
                marks=marks,
                module_hint=state.current_module,
                module_hint_is_explicit=state.module_is_explicit,
                year_context=year,
                qn_group=state.current_qn_group,
                raw_lines=[stripped],
            )
            state.flush()
            return

        # ── Plain text line ending with marks annotation ─────────────────
        marks_m = _RE_MARKS_LINE.match(stripped)
        if marks_m:
            state.flush()
            state.bullet_counter += 1
            year = _extract_year_from_line(stripped) or state.current_year
            clean_text = _strip_marks(marks_m.group(1)).strip()
            marks = int(marks_m.group(2))
            if not clean_text or len(clean_text) < 8:
                return
            if state.current_part == "B" and state.current_qn_group:
                q_num = f"{state.current_qn_group}b{state.bullet_counter}"
            else:
                q_num = str(state.bullet_counter)
            state.current_q = QuestionDTO(
                question_number=q_num,
                text=clean_text,
                part=state.current_part,
                marks=marks,
                module_hint=state.current_module,
                module_hint_is_explicit=state.module_is_explicit,
                year_context=year,
                qn_group=state.current_qn_group,
                raw_lines=[stripped],
            )
            state.flush()
            return

        # ── Standard numbered question: "1. text" ───────────────────────
        q_m = _RE_QUESTION_START.match(stripped)
        if q_m:
            state.flush()
            q_num = q_m.group(1)
            # Strip question number from text already done by group(2)
            q_text = _strip_marks(q_m.group(2)).strip()
            marks = _extract_marks(stripped)
            if marks is None:
                marks = 3 if state.current_part == "A" else 14
            year = _extract_year_from_line(stripped) or state.current_year
            state.current_q = QuestionDTO(
                question_number=q_num,
                text=q_text,
                part=state.current_part,
                marks=marks,
                module_hint=state.current_module,
                module_hint_is_explicit=state.module_is_explicit,
                year_context=year,
                qn_group=state.current_qn_group,
                raw_lines=[stripped],
            )
            return

        # ── Sub-question or continuation of numbered question ─────────────
        if state.current_q:
            sub_m = _RE_SUB_QUESTION.match(stripped)
            if sub_m:
                # Strip sub-question label from text
                sub_text = _strip_marks(sub_m.group(2)).strip()
                if sub_text and len(sub_text) >= 3:
                    state.current_q.sub_questions.append(sub_text)
                    state.current_q.raw_lines.append(stripped)
                    sub_marks = _extract_marks(stripped)
                    if sub_marks and not state.current_q.marks:
                        state.current_q.marks = sub_marks
                return
            # Continuation: append only if real text and not structural
            cont_text = _strip_marks(stripped).strip()
            if (
                cont_text
                and len(cont_text) >= 5
                and not _RE_PART_A.match(stripped)
                and not _RE_PART_B.match(stripped)
                and not _RE_MODULE.match(stripped)
                and not _RE_QN_MARKER.match(stripped)
                and not re.match(r"^\d{1,3}$", stripped)
            ):
                state.current_q.text += " " + cont_text
                state.current_q.raw_lines.append(stripped)

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------

    def persist(self, dtos: List[QuestionDTO], paper, module_map: dict) -> List:
        """
        Persist QuestionDTO objects to the database.

        Args:
            dtos:       List from :meth:`parse`.
            paper:      ``papers.Paper`` ORM instance.
            module_map: ``{module_number: Module ORM instance}``

        Returns:
            List of created ``questions.Question`` instances.
        """
        from apps.questions.models import Question

        created = []
        for dto in dtos:
            if not dto.is_valid():
                continue
            module_instance = module_map.get(dto.module_hint) if dto.module_hint else None

            years = []
            if dto.year_context:
                years = [dto.year_context]
            elif paper.year:
                years = [str(paper.year)]

            q = Question.objects.create(
                paper=paper,
                question_number=dto.question_number,
                # Store full_question_text for embedding (includes sub-questions)
                text=dto.full_question_text,
                sub_questions=dto.sub_questions,
                marks=dto.marks,
                part=dto.part,
                module=module_instance,
                years_appeared=years,
                # Store whether module was set from explicit heading
                module_manually_set=dto.module_hint_is_explicit,
            )
            created.append(q)

        logger.info("Persisted %d questions for paper '%s'", len(created), paper)
        return created
