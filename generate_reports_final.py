"""
generate_reports_final.py
=========================
Step 1 — Clean OCR garbage from every question in the database using Gemini.
Step 2 — Generate Module 1–5 PDF reports using the cleaned text.
Step 3 — Save the PDFs to the target directory.

Usage:
    python generate_reports_final.py

Output:
    d:\\S5\\Disaster Management\\Module 1.pdf
    d:\\S5\\Disaster Management\\Module 2.pdf
    ...
    d:\\S5\\Disaster Management\\Module 5.pdf
"""

import os
import sys
import json
import re
import django
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Django setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, str(BASE_DIR))
django.setup()

from django.conf import settings
from apps.questions.models import Question
from apps.subjects.models import Subject, Module
from apps.reports.ktu_report_generator import KTUModuleReportGenerator

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_DIR = Path(r'd:\S5\Disaster Management')
GEMINI_API_KEY = getattr(settings, 'GEMINI_API_KEY', '')
SUBJECT_CODE = 'MCN301'

# ── Gemini helper ─────────────────────────────────────────────────────────────
def _gemini_call(prompt: str) -> str | None:
    """Call Gemini API and return the response text, or None on failure."""
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=GEMINI_API_KEY)
        # Try flash-lite which has its own quota bucket
        for model in ('gemini-2.0-flash-lite', 'gemini-1.5-flash-latest',
                      'gemini-1.5-flash', 'gemini-2.0-flash'):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                continue
        return None
    except Exception as exc:
        return None


# ── Rule-based OCR cleaner (no API needed) ────────────────────────────────────
# Maps raw OCR text patterns → corrected question text.
# Handles the specific artefacts found in MCN301 papers.
_RULE_FIXES: list[tuple[str, str]] = [
    # Word-level fixes
    (r'\bIdentiff\b',           'Identify'),
    (r'\bIdentifu\b',           'Identify'),
    (r'\bdentifu\b',            'Identify'),
    (r'\bIl-xolain\b',         'Explain'),
    (r'\b-xolain\b',            'Explain'),
    (r'\bxolain\b',             'Explain'),
    (r'\bltaz-afi\b',           'hazard'),
    (r'\blltaz\b',              'hazard'),
    (r'\bItaz\b',               'hazard'),
    (r'\bpartiqipation\b',      'participation'),
    (r'\bpartiqipate\b',        'participate'),
    (r"t`eatures",              'features'),
    (r'!a\}[€}]r',             'layer'),
    (r'la\*\*r',                'layer'),
    (r'd[€}]taii',              'detail'),
    (r'd[€}]td[:\-]+',         'detail'),
    (r'Statethe\b',             'State the'),
    (r'hazardmapping\.',        'hazard mapping.'),
    (r'hazardmapping\b',        'hazard mapping'),
    (r'Monsoonand\b',           'Monsoon and'),
    (r'importanceofozone\b',    'importance of ozone'),
    # Trailing marks-count artefacts ("… 3" at end of sentence = "3 marks")
    (r'\s+3\s*$',               '.'),
    # Stray J characters appended by scanner
    (r'\s+J\s+J\s*',            ' '),
    (r'\s+J\s*$',               ''),
    (r'\s+J\s+(?=[A-Z])',       ' '),
    # What-are-their truncation
    (r'What are\'disasters\?',  'What are disasters?'),
    (r"What are'disasters\?",   'What are disasters?'),
]

# Complete-sentence fixes for questions that are truncated/missing key words
_COMPLETIONS: dict[str, str] = {
    'What are disasters? What are their':
        'What are disasters? What are their causes?',
    'Define hazard,, List different types of':
        'Define hazard. List different types of hazards.',
    'Explain the contemporary approaches to risk':
        'Explain the contemporary approaches to risk assessment.',
    "Define the term 'disaster":
        "Define the term 'disaster response'.",
    'What distinguishes crisis counselling from regular':
        'What distinguishes crisis counselling from regular counselling?',
    'Explain the importance of communication in disaster':
        'Explain the importance of communication in disaster management.',
    'List the structural and nonstructural measures in capacity':
        'List the structural and nonstructural measures in capacity building.',
    'What are the benefits of stakeholder participation in Disaster risk':
        'What are the benefits of stakeholder participation in Disaster risk management?',
    'Discuss the two types of monsoon in Indian':
        'Discuss the two types of monsoon in the Indian subcontinent.',
    'Explain the major reasons for ozone':
        'Explain the major reasons for ozone depletion.',
    'State and explain crisis counselling. Identify the necessity of crisis':
        'State and explain crisis counselling. Identify the necessity of crisis counselling.',
    'Explain the uses of':
        'Explain the uses of hazard maps.',
    'Explain disaster response and the major factors affecting disaster response':
        'Explain disaster response and the major factors affecting disaster response.',
    'How man made disaster are varied from natural disasters. Explain with':
        'How are man-made disasters varied from natural disasters? Explain with examples.',
    'Explain the role of National Institute of Disaster Management in our':
        'Explain the role of National Institute of Disaster Management in our country.',
    'Explain any three natural disasters with examples from our':
        'Explain any three natural disasters with examples from our country.',
    'Describe the effective ways of promoting Stakeholder':
        'Describe the effective ways of promoting Stakeholder engagement.',
    'Explain the guiding principles and priorities of action according to the Sendai':
        'Explain the guiding principles and priorities of action according to the Sendai Framework.',
    'Explain major contemporary approaches to disaster risk assessment':
        'Explain major contemporary approaches to disaster risk assessment.',
    'Identiff the standard operating procedures to be followed during a disaster stage':
        'Identify the standard operating procedures to be followed during a disaster stage.',
    'Explain Indian monsoon in d':
        'Explain Indian monsoon in detail.',
    'Explain Green House Effect in d':
        'Explain Green House Effect in detail.',
    'Define lithosphere. Explain its composition and t`eatures.':
        'Define lithosphere. Explain its composition and features.',
    'List any 6 public health services required in responding to':
        'List any 6 public health services required in responding to disasters.',
}


def _rule_clean(text: str) -> str:
    """Apply deterministic regex + completion rules to fix known OCR patterns."""
    cleaned = text.strip()

    # Check exact completions first (prefix match)
    for prefix, full in _COMPLETIONS.items():
        if cleaned.startswith(prefix) and len(cleaned) <= len(prefix) + 15:
            return full

    # Apply word-level regex fixes
    for pattern, replacement in _RULE_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


# ── Junk detector (mirrors ktu_report_generator._is_junk) ────────────────────
def _is_junk(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return True
    for pat in [
        r'^\s*[A-Z]\s+subcontinent',
        r'PART\s*[B8]\s*[\{\[]',
        r'ea€\*\s*module',
        r'^[\W\s]{0,3}$',
        r'ltaz[-\s]*afi',
        r'!a\}[€}]r',
    ]:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# ── OCR garbage detector ──────────────────────────────────────────────────────
def _needs_cleaning(text: str) -> bool:
    """Heuristic: does this text still have OCR artifacts?"""
    if _is_junk(text):
        return False  # Junk — Gemini can't help; skip
    garbage_signals = [
        r'[€@\$~\^]{1,}',                    # Unicode artifacts
        r'[a-z]`[a-z]',                       # backtick inside word
        r'\b\w*[!|]{1}\w+\b',                 # exclamation inside word
        r'\b[a-z]{1,2}[}\]}{][a-z]\b',        # e.g., !a}r, la}€r
        r'Identiff',
        r'\bxolain\b',
        r'\bdentifu\b',
        r'^\s*-\s*xolain',
        r'[A-Za-z]\{[A-Za-z]',               # e.g., ea€* → ea{
        r'[A-Za-z0-9]\s+3\s*$',              # trailing " 3" = marks artefact
        r'\bItaz\b',
        r'\blltaz\b',
    ]
    t = text.lower()
    for pat in garbage_signals:
        if re.search(pat, text, re.IGNORECASE):
            return True
    # Very short / truncated (real questions are usually 15+ chars)
    if len(text.strip()) < 20 and not text.strip().endswith('?'):
        return True
    return False


# ── Batch-clean questions in DB ───────────────────────────────────────────────
def clean_questions_in_db(subject: Subject) -> int:
    """
    For every question that has OCR garbage and no cleaned_text yet,
    call Gemini to produce a corrected version and save it.
    Returns count of questions cleaned.
    """
    all_qs = Question.objects.filter(paper__subject=subject).select_related('paper')

    # Identify questions that need cleaning
    to_clean = [
        q for q in all_qs
        if (not q.cleaned_text) and _needs_cleaning(q.text)
    ]

    print(f"\n{'─'*60}")
    print(f"  OCR CLEANING: {len(to_clean)} questions need Gemini cleaning")
    print(f"  ({all_qs.count()} total questions in database)")
    print(f"{'─'*60}")

    if not to_clean:
        print("  All questions already clean — skipping OCR step.\n")
        return 0

    BATCH_SIZE = 12
    cleaned_count = 0

    for batch_start in range(0, len(to_clean), BATCH_SIZE):
        batch = to_clean[batch_start: batch_start + BATCH_SIZE]
        print(f"\n  Batch {batch_start // BATCH_SIZE + 1} "
              f"({len(batch)} questions) ...", flush=True)

        # Build numbered list for the prompt
        numbered = "\n".join(
            f"{i + 1}. {q.text.strip()[:350]}"
            for i, q in enumerate(batch)
        )

        prompt = (
            'You are reconstructing OCR-extracted exam questions from KTU university '
            'question papers on "Disaster Management" (subject MCN301).\n\n'
            'The OCR has introduced garbled characters, truncated words, merged words, '
            'stray symbols, and marks-count artefacts (e.g. a trailing "3" meaning '
            '"3 marks").\n\n'
            'Fix each question below:\n'
            '- Correct garbled/misspelled words (e.g. "Identiff" → "Identify", '
            '"!a}€r" → "layer", "xolain" → "Explain", "ltaz-afi" → "hazard", '
            '"t`eatures" → "features", "dentifu" → "Identify", trailing "3" → remove)\n'
            '- Complete obviously truncated questions (end them naturally)\n'
            '- Remove stray OCR marks, page-header fragments, and artefacts\n'
            '- Preserve the original academic question intent exactly\n'
            '- If a line is a page header or completely unrecoverable garbage, return null\n\n'
            f'Questions:\n{numbered}\n\n'
            'Return ONLY a valid JSON array of strings (or null for unrecoverable), '
            'one entry per question, no commentary:\n'
            '["cleaned q1", "cleaned q2", null, ...]'
        )

        response = _gemini_call(prompt)
        results = []

        if response:
            try:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    results = json.loads(match.group())
            except Exception:
                pass

        # Pad if Gemini returned fewer items than batch size
        while len(results) < len(batch):
            results.append(None)

        # Save cleaned texts
        for q, cleaned in zip(batch, results):
            if cleaned and isinstance(cleaned, str) and len(cleaned.strip()) >= 10:
                q.cleaned_text = cleaned.strip()
                q.save(update_fields=['cleaned_text'])
                print(f"    ✓ {cleaned[:70]}...")
                cleaned_count += 1
            else:
                print(f"    ✗ Could not clean: {q.text[:50]!r}")

    print(f"\n  Cleaning complete — {cleaned_count} questions updated.\n")
    return cleaned_count


# ── Generate canonical topic names via Gemini ─────────────────────────────────
def enrich_topic_names(module_num: int, priority: dict) -> dict:
    """
    For each priority tier, use Gemini to replace the full question
    text used as a topic label with a 3–6 word canonical academic label.
    E.g. "Explain the guiding principles..." → "Sendai Framework Priorities"
    """
    all_items = (
        priority.get('top', []) +
        priority.get('high', []) +
        priority.get('medium', []) +
        priority.get('low', [])
    )
    if not all_items:
        return priority

    numbered = "\n".join(
        f"{i + 1}. {item['topic']}"
        for i, item in enumerate(all_items)
    )

    prompt = (
        f'Below are exam question topics from Module {module_num} of '
        f'"Disaster Management" (KTU MCN301).\n\n'
        f'For each, produce a precise academic canonical topic label of 3–6 words.\n'
        f'Examples:\n'
        f'  "Explain the guiding principles...Sendai Framework" → "Sendai Framework Priorities"\n'
        f'  "Define hazard. List different types..." → "Hazard Definition and Types"\n'
        f'  "What are the benefits of stakeholder participation..." → '
        f'"Stakeholder Participation Benefits"\n\n'
        f'Topics:\n{numbered}\n\n'
        f'Return ONLY a JSON array of short label strings, one per topic, in the same order:\n'
        f'["Label 1", "Label 2", ...]'
    )

    response = _gemini_call(prompt)
    labels = []
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                labels = json.loads(match.group())
        except Exception:
            pass

    while len(labels) < len(all_items):
        labels.append(None)

    # Apply labels back to items
    for item, label in zip(all_items, labels):
        if label and isinstance(label, str) and 2 < len(label) < 80:
            item['topic'] = label

    return priority


# ── Generate reports ──────────────────────────────────────────────────────────
def generate_reports(subject: Subject) -> None:
    """Run KTUModuleReportGenerator for all 5 modules, save to TARGET_DIR."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  GENERATING REPORTS — {subject.name}")
    print(f"  Output: {TARGET_DIR}")
    print(f"{'─'*60}\n")

    generator = KTUModuleReportGenerator(subject, output_dir=str(TARGET_DIR))

    # Monkey-patch the priority analysis to add Gemini topic naming after clustering
    _orig_analyze = generator._analyze_priorities_from_questions.__func__

    def _analyze_with_canonical(self, questions):
        priority = _orig_analyze(self, questions)
        # Find module number from the questions (cheap lookup)
        mod_num = 0
        qs = list(questions)
        if qs and qs[0].module:
            mod_num = qs[0].module.number
        if mod_num:
            print(f"    Enriching topic labels for Module {mod_num} via Gemini...",
                  flush=True)
            priority = enrich_topic_names(mod_num, priority)
        return priority

    import types
    generator._analyze_priorities_from_questions = types.MethodType(
        _analyze_with_canonical, generator
    )

    for module_num in range(1, 6):
        module = Module.objects.filter(subject=subject, number=module_num).first()
        if not module:
            print(f"  ✗ Module {module_num} not found in database")
            continue

        print(f"  Generating Module {module_num} ...", flush=True, end=' ')
        pdf_path = generator.generate_module_report(module)

        if pdf_path and Path(pdf_path).exists():
            print(f"✓  →  {pdf_path}")
        else:
            print("✗ FAILED")

    print(f"\n  Reports saved to: {TARGET_DIR}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  KTU Report Generator — Final Output")
    print("  Subject: Disaster Management (MCN301)")
    print("=" * 60)

    # Locate the subject
    subject = (
        Subject.objects.filter(code__icontains='MCN301').first()
        or Subject.objects.filter(name__icontains='Disaster').first()
        or Subject.objects.first()
    )

    if not subject:
        print("\n  ERROR: No subject found in the database.")
        print("  Make sure the papers have been uploaded and processed first.")
        sys.exit(1)

    print(f"\n  Subject found: {subject.name} ({subject.code})")

    # Step 1 — Clean OCR garbage
    clean_questions_in_db(subject)

    # Step 2 — Generate module PDFs
    generate_reports(subject)

    print("=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")
