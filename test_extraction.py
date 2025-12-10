"""Test question extraction."""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.analysis.views import ManualAnalyzeView
from apps.papers.models import Paper

paper = Paper.objects.first()
if not paper:
    print("No papers found!")
    exit(1)

print(f"Testing extraction on: {paper.title}")
print(f"Raw text length: {len(paper.raw_text)} chars")
print(f"\n{'='*80}\n")

view = ManualAnalyzeView()
questions = view._extract_ktu_questions_improved(paper.raw_text)

print(f"✓ Extracted {len(questions)} questions:\n")
for q in questions:
    text_preview = q['text'][:80].replace('\n', ' ')
    print(f"Q{q['question_number']:>2}: {text_preview}...")
    
print(f"\n{'='*80}")
print(f"Total: {len(questions)} questions extracted")
print(f"Expected: 20 questions")
print(f"Success rate: {len(questions)/20*100:.1f}%")
if len(questions) < 20:
    print(f"❌ Missing {20-len(questions)} questions!")
else:
    print(f"✅ All questions extracted successfully!")
