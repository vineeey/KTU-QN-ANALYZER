"""Analyze raw text format."""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.papers.models import Paper
import re

paper = Paper.objects.first()
text = paper.raw_text

print("=== RAW TEXT ANALYSIS ===\n")

# Look for all lines that start with a digit
lines = text.split('\n')
for i, line in enumerate(lines):
    line_stripped = line.strip()
    if line_stripped and line_stripped[0].isdigit():
        # Check if it looks like a question number
        match = re.match(r'^(\d{1,2})\s*(.{0,100})', line_stripped)
        if match:
            num = match.group(1)
            rest = match.group(2)
            print(f"Line {i:3d}: Q{num:>2} | {rest[:80]}")

print("\n=== SEARCHING FOR SPECIFIC PATTERNS ===\n")

# Test different patterns
patterns = [
    (r'(?m)^\s*(\d{1,2})\s+(.+)$', "Pattern 1: Number + text"),
    (r'(?m)^(\d{1,2})\s*(.+)$', "Pattern 2: Number at line start"),
    (r'\n(\d{1,2})\s+([A-Z].+?)(?=\n\d{1,2}\s+|\Z)', "Pattern 3: Number with lookahead"),
]

for pattern, desc in patterns:
    matches = re.findall(pattern, text)
    print(f"{desc}: Found {len(matches)} matches")
    for m in matches[:3]:
        print(f"  Q{m[0]}: {m[1][:50]}...")
