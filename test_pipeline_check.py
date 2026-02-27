#!/usr/bin/env python3
"""
Comprehensive pipeline verification script.
Tests: PaddleOCR engine → text cleaning → segmentation → classification.
"""
import os
import sys
import time
import logging
import glob

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(__file__))

import django
django.setup()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = []


def record(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {status}  {name}" + (f"  →  {detail}" if detail else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ======================================================================
# 1. IMPORT CHECKS
# ======================================================================
section("1. Import checks")

try:
    from paddleocr import PaddleOCR
    record("PaddleOCR import", True)
except ImportError as e:
    record("PaddleOCR import", False, str(e))

try:
    import fitz
    record("PyMuPDF (fitz) import", True)
except ImportError as e:
    record("PyMuPDF (fitz) import", False, str(e))

try:
    from PIL import Image
    record("Pillow import", True)
except ImportError as e:
    record("Pillow import", False, str(e))

try:
    import cv2
    record("OpenCV (cv2) import", True)
except ImportError as e:
    record("OpenCV (cv2) import", False, str(e))

try:
    import numpy as np
    record("NumPy import", True)
except ImportError as e:
    record("NumPy import", False, str(e))

# ======================================================================
# 2. OCR ENGINE
# ======================================================================
section("2. OCR Engine (PaddleOCR)")

try:
    from apps.analysis.services.ocr_engine import OCREngine, OCRResult

    # Reset singleton for fresh init
    OCREngine._ocr_loaded = False
    OCREngine._paddle_ocr = None

    t0 = time.time()
    engine = OCREngine()
    init_time = time.time() - t0
    record("OCREngine initialisation", OCREngine._ocr_loaded, f"{init_time:.1f}s")

    # Test with a real PDF
    pdfs = sorted(glob.glob("media/papers/*.PDF")) + sorted(glob.glob("media/papers/*.pdf"))
    if pdfs:
        test_pdf = pdfs[0]
        print(f"\n  Testing PDF: {os.path.basename(test_pdf)}")

        t0 = time.time()
        ocr_results = engine.process_pdf(test_pdf)
        ocr_time = time.time() - t0

        record("process_pdf returned results", len(ocr_results) > 0,
               f"{len(ocr_results)} pages in {ocr_time:.1f}s")

        if ocr_results:
            total_chars = sum(len(r.text) for r in ocr_results)
            avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
            record("Total chars extracted", total_chars > 50,
                   f"{total_chars} chars, avg confidence: {avg_conf:.2f}")

            # Show first page sample
            first = ocr_results[0]
            sample = first.text[:200].replace('\n', ' ↵ ')
            print(f"\n  --- Page 1 sample ({len(first.text)} chars, conf={first.confidence:.2f}) ---")
            print(f"  {sample}...")
            print(f"  --- End sample ---")

            # Show last page sample if multi-page
            if len(ocr_results) > 1:
                last = ocr_results[-1]
                sample = last.text[:200].replace('\n', ' ↵ ')
                print(f"\n  --- Page {last.page_number} sample ({len(last.text)} chars, conf={last.confidence:.2f}) ---")
                print(f"  {sample}...")
                print(f"  --- End sample ---")
    else:
        record("PDF test file available", False, "No PDFs in media/papers/")

except Exception as e:
    record("OCR Engine test", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# 3. TEXT CLEANER
# ======================================================================
section("3. Text Cleaner")

try:
    from apps.analysis.services.text_cleaner import (
        TextCleaner, normalize_whitespace, remove_common_artifacts,
        rule_based_corrections, clean_ktu_exam_artifacts
    )

    # Unit tests
    ws = normalize_whitespace("Hello   world\n\n  Foo   bar  ")
    record("normalize_whitespace", ws == "Hello world\n\nFoo bar", repr(ws))

    art = remove_common_artifacts("Question text\n- 3 -\nMore text")
    record("remove_common_artifacts (page number)", "- 3 -" not in art)

    art2 = remove_common_artifacts("Some text\nPage 2 of 10\nMore")
    record("remove_common_artifacts (page label)", "Page 2 of 10" not in art2)

    lig = rule_based_corrections("deﬁne the ﬂood risk")
    record("rule_based_corrections (ligatures)", "ﬁ" not in lig and "fi" in lig)

    spaced = rule_based_corrections("1 . Define hazard")
    record("rule_based_corrections (spaced Q number)", "1." in spaced)

    hyph = rule_based_corrections("compu-\nter science")
    record("rule_based_corrections (broken hyphen)", "computer" in hyph)

    ktu = clean_ktu_exam_artifacts("J\n3\nPART A\n1. Define disaster")
    record("clean_ktu_exam_artifacts", "Part A" in ktu and "Define disaster" in ktu)

    # Full pipeline
    cleaner = TextCleaner()
    raw = "  1 . Deﬁne disaster   \n- 3 -\nPage 1 of 5\nTurn over\n"
    cleaned = cleaner.clean(raw)
    record("TextCleaner.clean() full pipeline",
           "Page 1 of 5" not in cleaned and "Turn over" not in cleaned.lower(),
           f"'{cleaned[:80]}...'")

    # Test with actual OCR output if available
    if ocr_results:
        raw_page = ocr_results[0].text
        cleaned_page = cleaner.clean(raw_page)
        record("Clean actual OCR output",
               len(cleaned_page) > 0 and len(cleaned_page) <= len(raw_page) * 1.1,
               f"raw={len(raw_page)} → cleaned={len(cleaned_page)} chars")

except Exception as e:
    record("Text Cleaner test", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# 4. SEGMENTER
# ======================================================================
section("4. Question Segmenter")

try:
    from apps.analysis.services.segmenter import Segmenter, QuestionDTO

    seg = Segmenter()

    # Test basic Part A extraction
    text1 = "PART A\n1. Define disaster.\n2. What is hazard?\n3. Explain vulnerability."
    dtos1 = seg.parse(text1)
    record("Basic Part A parsing", len(dtos1) >= 2,
           f"Extracted {len(dtos1)} questions")

    if dtos1:
        record("Questions have part='A'", all(d.part == "A" for d in dtos1))
        record("Question text is valid", all(d.is_valid() for d in dtos1))

    # Test Part B with marks
    text2 = "Part B\n11. Explain disaster management cycle in detail. (10 marks)"
    dtos2 = seg.parse(text2)
    record("Part B parsing", len(dtos2) >= 1,
           f"Extracted {len(dtos2)} questions")
    if dtos2:
        record("Part B marks extracted", dtos2[0].marks is not None,
               f"marks={dtos2[0].marks}")

    # Test with actual cleaned OCR text
    if ocr_results:
        full_text = "\n\n".join(
            cleaner.clean(r.text) for r in ocr_results
        )
        real_dtos = seg.parse(full_text)
        record("Segment real OCR text",
               len(real_dtos) >= 1,
               f"Extracted {len(real_dtos)} questions from real PDF")

        if real_dtos:
            print(f"\n  --- Extracted questions preview ---")
            for i, dto in enumerate(real_dtos[:8]):
                txt_preview = dto.text[:80].replace('\n', ' ')
                print(f"  Q{dto.question_number} (Part {dto.part}, "
                      f"marks={dto.marks}): {txt_preview}")
            if len(real_dtos) > 8:
                print(f"  ... and {len(real_dtos) - 8} more questions")
            print(f"  --- End preview ---")

except Exception as e:
    record("Segmenter test", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# 5. CLASSIFIER
# ======================================================================
section("5. Module Classifier")

try:
    from apps.analysis.services.classifier import ModuleClassifier
    from unittest.mock import MagicMock

    def _make_module(number, name, keywords=None):
        m = MagicMock()
        m.number = number
        m.name = name
        m.keywords = keywords or []
        m.topics = []
        m.description = name
        return m

    clf = ModuleClassifier()
    record("ModuleClassifier initialised", True)

    modules = [
        _make_module(1, "Introduction to Disasters", ["disaster", "hazard", "vulnerability"]),
        _make_module(2, "Disaster Mitigation", ["mitigation", "preparedness", "response"]),
        _make_module(3, "Disaster Policy", ["NDMA", "policy", "framework", "act"]),
        _make_module(4, "Recovery", ["recovery", "rehabilitation", "reconstruction"]),
    ]

    # Stage 1 keyword test
    r1 = clf._stage1_keywords("Define disaster and explain its types", modules)
    record("Stage 1 keyword match", r1 == 1, f"module={r1}")

    r2 = clf._stage1_keywords("Explain mitigation strategies", modules)
    record("Stage 1 keyword match (module 2)", r2 == 2, f"module={r2}")

    r3 = clf._stage1_keywords("Discuss the role of NDMA in policy", modules)
    record("Stage 1 keyword match (module 3)", r3 == 3, f"module={r3}")

    # Module hint priority
    subject = MagicMock()
    r4 = clf.classify("Random text", subject, modules, module_hint=4)
    record("Module hint takes priority", r4 == 4, f"module={r4}")

except Exception as e:
    record("Classifier test", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# 6. IMAGE PREPROCESSOR
# ======================================================================
section("6. Image Preprocessor")

try:
    from apps.analysis.services.image_preprocessor import (
        ImagePreprocessor, to_grayscale, apply_median_blur,
        apply_sharpen_filter, adaptive_gaussian_threshold, morphological_closing
    )

    # Create a test image (white background with black text)
    test_img = Image.new("RGB", (200, 50), "white")
    record("ImagePreprocessor imports", True)

    gray = to_grayscale(test_img)
    record("to_grayscale", gray.mode == "L", f"mode={gray.mode}")

    blurred = apply_median_blur(gray)
    record("apply_median_blur", blurred.size == gray.size)

    sharpened = apply_sharpen_filter(gray)
    record("apply_sharpen_filter", sharpened.size == gray.size)

    binary = adaptive_gaussian_threshold(gray)
    record("adaptive_gaussian_threshold", binary.size == gray.size)

    closed = morphological_closing(gray)
    record("morphological_closing", closed.size == gray.size)

    # Full preprocessor class
    pp = ImagePreprocessor()
    enhanced = pp.enhance_for_ocr(test_img)
    record("ImagePreprocessor.enhance_for_ocr", enhanced is not None)

except Exception as e:
    record("Image Preprocessor test", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# 7. END-TO-END PDF PIPELINE
# ======================================================================
section("7. End-to-End Pipeline (PDF → Questions)")

try:
    if pdfs:
        test_pdf = pdfs[0]
        print(f"\n  Full pipeline on: {os.path.basename(test_pdf)}")

        # Step 1-3: OCR
        t0 = time.time()
        engine2 = OCREngine()
        ocr_out = engine2.process_pdf(test_pdf)

        # Step 4: Cleaning
        cleaner2 = TextCleaner()
        cleaned_parts = [cleaner2.clean(r.text) for r in ocr_out]
        full_text = "\n\n".join(cleaned_parts)

        # Step 5: Segmentation
        seg2 = Segmenter()
        questions = seg2.parse(full_text)

        # Step 6: Classification (with mock modules)
        clf2 = ModuleClassifier()
        classified = 0
        for q in questions:
            if q.module_hint:
                classified += 1
            else:
                mod = clf2._stage1_keywords(q.text, modules)
                if mod:
                    q.module_hint = mod
                    classified += 1

        elapsed = time.time() - t0

        record("End-to-end pipeline",
               len(questions) >= 1,
               f"{len(questions)} questions, {classified} classified, {elapsed:.1f}s total")

        # Summary
        part_a = [q for q in questions if q.part == "A"]
        part_b = [q for q in questions if q.part == "B"]
        print(f"\n  Pipeline Results:")
        print(f"    Pages OCR'd:       {len(ocr_out)}")
        print(f"    Total chars:       {sum(len(r.text) for r in ocr_out)}")
        print(f"    After cleaning:    {len(full_text)} chars")
        print(f"    Questions found:   {len(questions)}")
        print(f"    Part A:            {len(part_a)}")
        print(f"    Part B:            {len(part_b)}")
        print(f"    Classified:        {classified}/{len(questions)}")
        print(f"    Time:              {elapsed:.1f}s")
    else:
        record("End-to-end pipeline", False, "No test PDFs available")

except Exception as e:
    record("End-to-end pipeline", False, str(e))
    import traceback; traceback.print_exc()


# ======================================================================
# SUMMARY
# ======================================================================
section("SUMMARY")

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total = len(results)

print(f"\n  {PASS} Passed: {passed}/{total}")
if failed:
    print(f"  {FAIL} Failed: {failed}/{total}")
    print(f"\n  Failed tests:")
    for name, p, detail in results:
        if not p:
            print(f"    {FAIL} {name}: {detail}")
else:
    print(f"\n  All tests passed! Pipeline is working correctly.")

print()
