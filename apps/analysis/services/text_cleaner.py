"""
Strict OCR text cleaning for KTU exam paper analysis.

Pipeline order (must run before any embedding or segmentation):
  1. normalize_whitespace         – collapse spaces/tabs, preserve paragraph breaks
  2. remove_university_headers    – strip institution headers, reg no, exam metadata
  3. remove_marks_annotations     – strip (3 marks), (7), [10], Marks: 5, etc.
  4. remove_common_artifacts      – page numbers, watermarks, P.T.O., stray lines
  5. rule_based_corrections       – ligatures, broken hyphenated words
  6. merge_broken_words           – rejoin OCR-split words ("confi guration")
  7. remove_isolated_ocr_noise    – drop lone letters/digits/symbols that are OCR junk
  8. clean_ktu_exam_artifacts      – KTU-specific garbled headers, question numbers
  9. contextual_hook_stub         – optional LLM hook (fail-safe, never summarises)

CRITICAL RULES enforced here:
- NEVER summarise, rephrase, or change the meaning of questions.
- NEVER remove structural markers (Part A, Part B, Module N, year headers).
- NEVER let raw OCR text reach the embedding stage.
- Question boundary lines are always preserved.
"""

from __future__ import annotations

import re
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at module load)
# ---------------------------------------------------------------------------

# ── University / institution header lines ──────────────────────────────────
_RE_UNIV_HEADERS = re.compile(
    r"(?im)^.*?"
    r"(?:"
    r"APJ\s*abdul\s*kalam|"
    r"technological\s+university|"
    r"b\.?tech\.?\s+degree\s+examination|"
    r"b\.?e\.?\s+degree\s+examination|"
    r"m\.?tech\.?\s+degree|"
    r"reg(?:istration)?\s*no\.?:?|"
    r"roll\s*no\.?:?|"
    r"max(?:imum)?\s*(?:time|marks)\s*:?|"
    r"time\s*:\s*\d|"
    r"duration\s*:\s*\d|"
    r"instructions?\s+to\s+(?:candidates?|students?)|"
    r"answer\s+all\s+questions|"
    r"answer\s+any\s+(?:\w+)\s+questions|"
    r"course\s+code\s*:?|"
    r"course\s+name\s*:?|"
    r"semester\s*:?\s*\d|"
    r"branch\s*:?|"
    r"use\s+of.*?permitted|"
    r"assume\s+suitable\s+data"
    r").*?$",
    re.IGNORECASE,
)

# ── Marks annotations embedded inside or at end of question text ───────────
# Patterns:  (3 marks)  (7 Marks)  [10]  [10 marks]  Marks: 5  (3×2=6 marks)
# Also:  – (Dec 2022, 3 marks)  or  - (3 marks)
_RE_MARKS_ANNOTATION = re.compile(
    r"""
    (?:
        # Pattern: dash/em-dash followed by (month year, N marks)
        \s*[—–\-]\s*
        \(\s*[A-Za-z]{3,9}\.?\s+20\d{2}\s*,\s*\d+\s*marks?\s*\)
    |
        # Pattern: (N marks)  or  (N×M=P marks)
        \(\s*\d+\s*(?:[×xX\*]\s*\d+\s*=\s*\d+\s*)?marks?\s*\)
    |
        # Pattern: [N marks]  or  [N]  (only if standalone bracket-number)
        \[\s*\d+\s*(?:marks?)?\s*\]
    |
        # Pattern: Marks: N  or  marks = N
        \bmarks?\s*[:=]\s*\d+\b
    |
        # Pattern: (N) at end of line — bare number in parens
        \(\s*\d{1,2}\s*\)\s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Page numbers and navigational footers ─────────────────────────────────
_RE_PAGE_ARTIFACTS = [
    re.compile(r"(?im)^[-–—]*\s*\d{1,3}\s*[-–—]*$"),            # –3–  or  3
    re.compile(r"(?im)^(page|pg\.?)\s+\d+(\s+of\s+\d+)?$"),     # Page 2 of 10
    re.compile(r"(?im)^(turn\s+over|contd\.?|p\.t\.o\.?)$"),     # P.T.O.
    re.compile(r"(?im)^\s*[|_]{2,}\s*$"),                         # table borders
    re.compile(r"(?im)^[^\w\s]$"),                                 # lone punctuation line
]

# ── Common broken-word patterns seen in PaddleOCR output ──────────────────
# "confi guration" → "configuration",  "algo rithm" → "algorithm"
# Strategy: if a word-fragment ends with a space then continues with a
# lowercase letter, AND the result is a known tech word → merge.
_BROKEN_WORD_PAIRS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bconfi\s+guration\b",  re.I), "configuration"),
    (re.compile(r"\balgo\s+rithm\b",       re.I), "algorithm"),
    (re.compile(r"\bimple\s+mentation\b",  re.I), "implementation"),
    (re.compile(r"\bsoftware\s+(?=en)",    re.I), "software "),  # keep space
    (re.compile(r"\bde\s+fi\s*ne\b",       re.I), "define"),
    (re.compile(r"\bdefi\s+ne\b",          re.I), "define"),
    (re.compile(r"\bana\s+lysis\b",        re.I), "analysis"),
    (re.compile(r"\barchitec\s+ture\b",    re.I), "architecture"),
    (re.compile(r"\bspecifi\s+cation\b",   re.I), "specification"),
    (re.compile(r"\bverifi\s+cation\b",    re.I), "verification"),
    (re.compile(r"\bcertifi\s+cate\b",     re.I), "certificate"),
    (re.compile(r"\bmanage\s+ment\b",      re.I), "management"),
    (re.compile(r"\bdevelop\s+ment\b",     re.I), "development"),
    (re.compile(r"\brequire\s+ment\b",     re.I), "requirement"),
    (re.compile(r"\bdiagram\s+matic\b",    re.I), "diagrammatic"),
    (re.compile(r"\binforma\s+tion\b",     re.I), "information"),
    (re.compile(r"\bcommuni\s+cation\b",   re.I), "communication"),
    (re.compile(r"\bintegra\s+tion\b",     re.I), "integration"),
    (re.compile(r"\bcompati\s+bility\b",   re.I), "compatibility"),
    (re.compile(r"\bdis\s+aster\b",        re.I), "disaster"),
    (re.compile(r"\bevalu\s+ation\b",      re.I), "evaluation"),
    (re.compile(r"\bplanning\s+(?=and)\b", re.I), "planning "),
    (re.compile(r"\bplan\s+ning\b",        re.I), "planning"),
    (re.compile(r"\btest\s+ing\b",         re.I), "testing"),
    (re.compile(r"\bfunction\s+al\b",      re.I), "functional"),
    (re.compile(r"\bnon\s*-\s*function",   re.I), "non-function"),
    (re.compile(r"\bqualit\s+y\b",         re.I), "quality"),
    (re.compile(r"\bprocess\s+ing\b",      re.I), "processing"),
    (re.compile(r"\bresource\s+s\b",       re.I), "resources"),
    (re.compile(r"\bproject\s+s\b",        re.I), "projects"),
]

# ── Ligature and typographic normalisation ────────────────────────────────
_LIGATURES: dict[str, str] = {
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb00": "ff",   # ﬀ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00b0": " degrees ",
}

# ── Structural lines that must NEVER be removed ────────────────────────────
_RE_PRESERVE_STRUCTURAL = re.compile(
    r"^(?:"
    r"Part\s*[AB]|"
    r"Module\s+\d|"
    r"Unit\s+[IVX\d]+|"
    r"Qn\.?\s*\d|"
    r"Q\.?\s*\d|"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+20\d{2}"
    r")",
    re.IGNORECASE,
)

# ── Isolated OCR noise: lines with only 1–3 non-word characters or lone
# alphabetic chars that are clearly OCR misreads (J, I, !, 3, etc.) ────────
_RE_LONE_CHAR_LINE = re.compile(r"^[\W\d]{1,4}$")     # only punct/digits, len ≤ 4
_RE_SINGLE_CAPITAL = re.compile(r"^[A-Z]$")            # single capital letter
_RE_BARE_PAGE_NUMBER = re.compile(r"^\d{1,3}$")        # digits only (page numbers)


# ---------------------------------------------------------------------------
# Individual cleaning functions (pure, independently testable)
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs into a single space and strip
    leading/trailing whitespace from every line.  Preserves paragraph
    breaks (blank lines) so that section boundaries survive.

    >>> normalize_whitespace("Hello   world\\n\\n  Foo   bar  ")
    'Hello world\\n\\nFoo bar'
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned.append(line)
    # Collapse runs of more than 2 blank lines into 2
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def remove_university_headers(text: str) -> str:
    """
    Remove university/institution header lines that appear at the top of
    KTU exam papers (and sometimes as running headers on each page).

    Removed patterns include:
    - "APJ Abdul Kalam Technological University"
    - "B.Tech Degree Examination"
    - "Reg No:", "Roll No:", "Course Code:"
    - "Maximum Marks:", "Time:", "Duration:"
    - "Instructions to candidates", "Answer all questions"
    - Any line that only contains one of these markers

    Structural lines (Part A, Module 1, year headers) are NEVER removed.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        # Always preserve structural markers
        if _RE_PRESERVE_STRUCTURAL.match(s):
            cleaned.append(line)
            continue
        # Remove if matches university header pattern
        if s and _RE_UNIV_HEADERS.match(s):
            logger.debug("Removed university header: %r", s[:60])
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def remove_marks_annotations(text: str) -> str:
    """
    Strip marks annotations from question text so they do not pollute
    embeddings.  Preserves surrounding text and question structure.

    Removed:
    - (3 marks)  (7 Marks)  (3×2=6 marks)
    - [10]  [10 marks]
    - Marks: 5  marks = 7
    - – (Dec 2022, 3 marks) at end of line
    - (7) at end-of-line (bare single/double-digit in parens)

    Does NOT remove:
    - Question numbers like (a), (b), (i), (ii)
    - Year annotations that are part of a section header
    """
    # Remove full trailing date+marks annotation: " — (Dec 2022, 7 marks)"
    text = re.sub(
        r"\s*[—–\-]\s*\(\s*[A-Za-z]{3,9}\.?\s+20\d{2}\s*,\s*\d+\s*marks?\s*\)\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Remove (N marks) or (N×M=P marks)
    text = re.sub(
        r"\(\s*\d+\s*(?:[×xX\*]\s*\d+\s*=\s*\d+\s*)?marks?\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove [N marks] or [N]
    text = re.sub(r"\[\s*\d+\s*(?:marks?)?\s*\]", "", text, flags=re.IGNORECASE)
    # Remove Marks: N
    text = re.sub(r"\bmarks?\s*[:=]\s*\d+\b", "", text, flags=re.IGNORECASE)
    # Remove bare (N) at end of line (single/double digit only)
    text = re.sub(r"\(\s*\d{1,2}\s*\)\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_common_artifacts(text: str) -> str:
    """
    Remove typical OCR artefacts that appear in scanned exam papers:
    page numbers, watermarks, P.T.O., stray punctuation lines.

    Structural markers are never touched here.
    """
    for pat in _RE_PAGE_ARTIFACTS:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rule_based_corrections(text: str) -> str:
    """
    Deterministic character-level OCR substitution rules.

    Applied substitutions:
    - Ligature normalisation (ﬁ→fi, ﬂ→fl, etc.)
    - Spaced question numbers: "1 ." → "1."
    - Broken hyphenated words at line ends: "compu-\\nter" → "computer"
    - Roman numeral parens: "( i )" → "(i)"
    - Common OCR misreads: "1et" → "let", "1ist" → "list"

    Never rewrites semantic meaning.
    """
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)

    # Spaced question numbers
    text = re.sub(r"(\b\d{1,2})\s+\.", r"\1.", text)

    # Broken hyphenated words at line ends
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Roman numerals in parens
    text = re.sub(r"\(\s+([ivxlcdm]+)\s+\)", r"(\1)", text, flags=re.IGNORECASE)

    # Confident OCR letter-digit swaps
    text = re.sub(r"\b1et\b", "let", text)
    text = re.sub(r"\b1ist\b", "list", text)
    text = re.sub(r"\bI\b(?=\s+[a-z])", "1", text)  # lone "I" before lowercase → "1"

    return text


def merge_broken_words(text: str) -> str:
    """
    Merge OCR-split words back into correct single tokens.

    Handles patterns like:
    - "confi guration"  → "configuration"
    - "algo rithm"      → "algorithm"
    - "defi ne"         → "define"
    - "non -functional" → "non-functional"

    Only applies known substitutions to avoid false positives.
    """
    for pattern, replacement in _BROKEN_WORD_PAIRS:
        text = pattern.sub(replacement, text)
    return text


def remove_isolated_ocr_noise(text: str) -> str:
    """
    Remove lines that consist entirely of isolated OCR noise:
    - A single capital letter (J, I, O, etc.)
    - 1–4 characters that are all punctuation/digits
    - A bare page-number digit string

    Lines that are structural markers (Part A, Module 1, Qn 11, years) are
    always preserved and never removed.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        # Always preserve structural markers
        if _RE_PRESERVE_STRUCTURAL.match(s):
            cleaned.append(line)
            continue
        # Drop single capital letters (OCR noise: "J", "I")
        if _RE_SINGLE_CAPITAL.fullmatch(s):
            logger.debug("Removed isolated capital letter artifact: %r", s)
            continue
        # Drop bare page numbers
        if _RE_BARE_PAGE_NUMBER.fullmatch(s):
            logger.debug("Removed bare page number: %r", s)
            continue
        # Drop lines with only 1–4 non-word/digit characters
        if _RE_LONE_CHAR_LINE.fullmatch(s):
            logger.debug("Removed lone-char artifact line: %r", s)
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def clean_ktu_exam_artifacts(text: str) -> str:
    """
    Remove and correct OCR artefacts specific to KTU-style scanned exam
    papers (especially post-PaddleOCR output from re-rendered image pages).

    Handles:
    - Garbled Part A / Part B headers ("PARTA", "Part— A").
    - Question numbers split across lines ("1 .  question" → "1. question").
    - Runs of non-word chars that confuse the segmenter.
    - Stray leading/trailing symbols.
    """
    lines_in = text.splitlines()
    lines_out: list[str] = []

    for line in lines_in:
        s = line.strip()

        if not s:
            lines_out.append("")
            continue

        # Normalise garbled Part headers (must run before noise removal)
        s = re.sub(r"(?i)\bpart\s*[:\-—]?\s*a\b", "Part A", s)
        s = re.sub(r"(?i)\bpart\s*[:\-—]?\s*b\b", "Part B", s)

        # Fix space-separated question number at line start: "1 . " → "1. "
        s = re.sub(r"^(\d{1,2})\s+\.\s+", r"\1. ", s)

        # Strip stray leading/trailing symbols (not bullet chars •●¢–)
        s = re.sub(r"^[\*\+\=\|\\#@~`]+\s*", "", s)
        s = re.sub(r"\s*[\*\+\=\|\\#@~`]+$", "", s).strip()

        if s:
            lines_out.append(s)

    result = "\n".join(lines_out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def contextual_hook_stub(
    text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Optional final hook for LLM-based contextual correction.

    Contract:
    - If *llm_fn* is provided it is called with current text and its
      return value replaces it ONLY if the LLM returns non-empty text.
    - On any failure the original text is returned unchanged (fail-safe).
    - The LLM MUST NOT summarise, rephrase, or reorder questions.
      It is only permitted to fix remaining OCR character-level errors.
    - This module stays decoupled from any specific LLM provider.
    """
    if llm_fn is None:
        return text
    try:
        result = llm_fn(text)
        if isinstance(result, str) and result.strip():
            return result
        logger.debug("LLM contextual hook returned empty — keeping original text")
        return text
    except Exception as exc:
        logger.warning("LLM contextual hook failed (%s) — keeping original text", exc)
        return text


# ---------------------------------------------------------------------------
# Convenience pipeline class
# ---------------------------------------------------------------------------


class TextCleaner:
    """
    Applies the full strict cleaning pipeline in a single call.

    Pipeline order:
      1. normalize_whitespace
      2. remove_university_headers
      3. remove_marks_annotations
      4. remove_common_artifacts
      5. rule_based_corrections
      6. merge_broken_words
      7. remove_isolated_ocr_noise
      8. clean_ktu_exam_artifacts
      9. contextual_hook_stub   (optional LLM hook, fail-safe)

    Usage::

        cleaner = TextCleaner()
        clean = cleaner.clean(raw_ocr_text)

        # With optional LLM hook (character-level fixes only, no rephrasing):
        clean = cleaner.clean(raw_ocr_text, llm_fn=my_llm_correct)
    """

    def clean(
        self,
        text: str,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """
        Run the full strict cleaning pipeline.

        Args:
            text:    Raw OCR text from PaddleOCR (MUST be provided here,
                     NEVER passed directly to the embedder without cleaning).
            llm_fn:  Optional LLM correction callable ``(text: str) -> str``.
                     Called last; must preserve all question content exactly.

        Returns:
            Cleaned text ready for segmentation and embedding.
        """
        text = normalize_whitespace(text)
        text = remove_university_headers(text)
        text = remove_marks_annotations(text)
        text = remove_common_artifacts(text)
        text = rule_based_corrections(text)
        text = merge_broken_words(text)
        text = remove_isolated_ocr_noise(text)
        text = clean_ktu_exam_artifacts(text)
        text = contextual_hook_stub(text, llm_fn)
        return text
