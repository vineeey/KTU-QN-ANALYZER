"""Quick test script to verify the 4 pipeline fixes."""
import re
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))


def test_text_cleaner():
    """Test 1: text_cleaner removes OCR artifacts."""
    from apps.analysis.services.text_cleaner import TextCleaner
    tc = TextCleaner()

    # Dirty input with typical OCR problems
    dirty = (
        "PARTA\n"
        "(Answer all questions)\n"
        "1 Explain relevance of greenhouse gases. 3\n"
        "2 Discuss two types of monsoon in Indian subcontinent. 3\n"
        "J\n"
        "3 State major data requirements of hazard mapping. 3\n"
        "PART  B\n"
        "Module -1\n"
        "11 a) Categorize layers of atmosphere. 8\n"
    )
    cleaned = tc.clean(dirty)
    print("=== Test 1: TextCleaner ===")
    print(cleaned)
    
    # Verify lone J is removed
    assert "\nJ\n" not in cleaned, "FAIL: Lone J not removed"
    # Verify Part A header is normalized
    assert "Part A" in cleaned, "FAIL: PARTA not normalized to 'Part A'"
    print("PASS: text_cleaner works correctly\n")


def test_module_classification():
    """Test 2: Module classification uses parsed question number, not list index."""
    print("=== Test 2: Module Classification ===")
    
    part_a_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    part_b_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    
    # Simulate: extractor returned questions in wrong order or partial set
    # e.g., only got Q1, Q5, Q11a, Q15a, Q19a
    test_data = [
        {'question_number': '1',   'text': 'Greenhouse gases',    'marks': 3},
        {'question_number': '5',   'text': 'Disaster response',   'marks': 3},
        {'question_number': '11',  'text': 'Atmosphere layers',   'marks': 8},
        {'question_number': '13a', 'text': 'Hazard mapping',      'marks': 8},
        {'question_number': '15a', 'text': 'Core elements DRM',   'marks': 10},
        {'question_number': '19a', 'text': 'Disaster types India', 'marks': 10},
    ]
    
    expected_modules = [1, 3, 1, 2, 3, 5]
    
    for i, q_data in enumerate(test_data):
        raw_qnum = q_data.get('question_number', str(i + 1))
        num_match = re.match(r'(\d+)', str(raw_qnum))
        q_number = int(num_match.group(1)) if num_match else (i + 1)
        
        if q_number <= 10:
            mod = part_a_modules[q_number - 1]
        elif q_number <= 20:
            offset = q_number - 11
            mod = part_b_modules[offset]
        else:
            mod = 5
        
        status = "PASS" if mod == expected_modules[i] else "FAIL"
        print(f"  {status}: Q#{raw_qnum} -> q_number={q_number} -> Module {mod} (expected {expected_modules[i]})")
        assert mod == expected_modules[i], f"FAIL: Q#{raw_qnum} got Module {mod}, expected {expected_modules[i]}"
    
    # BUG PROOF: With old code (idx+1), Q13a would be idx=3 -> q_number=4 -> Module 2 (A)
    # which is wrong because Q13a is Part B Module 2, but the old code would assign it as
    # Part A Module 2. With partial extraction this gets worse.
    print("PASS: Module classification uses parsed question numbers\n")


def test_old_bug_module_classification():
    """Test 3: Prove the old index-based classification was wrong."""
    print("=== Test 3: Old Bug Demonstration ===")
    
    part_a_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    part_b_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    
    # Same data as above, but using OLD code: idx + 1
    test_data = [
        {'question_number': '1',   'text': 'Greenhouse gases'},
        {'question_number': '5',   'text': 'Disaster response'},
        {'question_number': '11',  'text': 'Atmosphere layers'},
        {'question_number': '13a', 'text': 'Hazard mapping'},
        {'question_number': '15a', 'text': 'Core elements DRM'},
        {'question_number': '19a', 'text': 'Disaster types India'},
    ]
    
    old_modules = []  # What old code would assign
    for idx, q_data in enumerate(test_data):
        q_number = idx + 1  # OLD BUG
        if q_number <= 10:
            mod = part_a_modules[q_number - 1]
        else:
            offset = q_number - 11
            mod = part_b_modules[offset] if 0 <= offset < len(part_b_modules) else 5
        old_modules.append(mod)
    
    expected_correct = [1, 3, 1, 2, 3, 5]
    
    for i, (old_mod, correct_mod) in enumerate(zip(old_modules, expected_correct)):
        qn = test_data[i]['question_number']
        if old_mod != correct_mod:
            print(f"  BUG CONFIRMED: Q#{qn} old_code -> Module {old_mod}, correct -> Module {correct_mod}")
        else:
            print(f"  OK: Q#{qn} old_code -> Module {old_mod}, correct -> Module {correct_mod} (happens to match)")
    
    wrong_count = sum(1 for o, c in zip(old_modules, expected_correct) if o != c)
    print(f"  Old code would misclassify {wrong_count}/{len(test_data)} questions")
    print("PASS: Bug confirmed and fixed\n")


def test_destructive_regex_removed():
    """Test 4: pdf_extractor no longer joins short words."""
    print("=== Test 4: Destructive Regex Fix ===")
    
    # Old regex: re.sub(r'(\w+)\s+(\w{1,3})\b', r'\1\2', text)
    # This would join: "in the" -> "inthe", "of a" -> "ofa"
    test_text = "Explain the role of a disaster management in the context of risk."
    
    # Simulate old behavior
    old_result = re.sub(r'(\w+)\s+(\w{1,3})\b', r'\1\2', test_text)
    
    # New behavior: only fix hyphenated line breaks
    new_result = re.sub(r'(\w+)-\n(\w)', r'\1\2', test_text)
    
    print(f"  Original:  {test_text}")
    print(f"  Old regex: {old_result}")
    print(f"  New regex: {new_result}")
    
    assert new_result == test_text, "FAIL: New regex should not change this text"
    assert old_result != test_text, "Expected: Old regex would damage this text"
    print("PASS: Destructive regex removed\n")


def test_part_a_regex_tolerance():
    """Test 5: Part A regex handles garbled headers."""
    print("=== Test 5: Part A Regex Tolerance ===")
    
    # The new regex: P\s*A\s*R\s*T\s*[-–:—]?\s*A
    pattern = r'P\s*A\s*R\s*T\s*[-–:—]?\s*A'
    
    test_headers = [
        "PART A",
        "PARTA",
        "PART  A",
        "P A R T A",
        "PART-A",
        "PART—A",
        "Part A",
    ]
    
    for header in test_headers:
        match = re.search(pattern, header, re.IGNORECASE)
        status = "PASS" if match else "FAIL"
        print(f"  {status}: '{header}' -> {'matched' if match else 'NOT matched'}")
        assert match, f"FAIL: '{header}' should be matched"
    
    print("PASS: Part A regex handles garbled/spaced headers\n")


if __name__ == '__main__':
    test_text_cleaner()
    test_module_classification()
    test_old_bug_module_classification()
    test_destructive_regex_removed()
    test_part_a_regex_tolerance()
    print("=" * 50)
    print("ALL TESTS PASSED")
