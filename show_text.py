"""Analyze raw text format with context."""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.papers.models import Paper

paper = Paper.objects.first()
text = paper.raw_text

print("=== FULL RAW TEXT ===\n")
print(text)
print("\n" + "="*80)
