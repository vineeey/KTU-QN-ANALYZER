"""
Test script to diagnose PDF extraction issues.
Run this to see actual text extracted from PDFs.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.papers.models import Paper
from PyPDF2 import PdfReader

def test_extraction():
    """Test PDF extraction on uploaded papers."""
    
    # Get all papers
    papers = Paper.objects.all().order_by('-created_at')
    
    if not papers.exists():
        print("No papers found in database.")
        print("Please upload papers first through the web interface.")
        return
    
    print(f"Found {papers.count()} papers in database\n")
    print("=" * 80)
    
    for paper in papers[:3]:  # Test first 3 papers
        print(f"\nPaper: {paper.title}")
        print(f"File: {paper.file.name}")
        
        if not paper.file:
            print("  ERROR: No file attached")
            continue
        
        try:
            # Try to read the PDF
            pdf_path = paper.file.path
            print(f"  Path: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                print(f"  ERROR: File not found at {pdf_path}")
                continue
            
            reader = PdfReader(pdf_path)
            print(f"  Pages: {len(reader.pages)}")
            
            # Extract text from first 3 pages
            full_text = ""
            for page_num in range(min(3, len(reader.pages))):
                page_text = reader.pages[page_num].extract_text()
                full_text += page_text
                print(f"\n  --- Page {page_num + 1} (first 500 chars) ---")
                print(page_text[:500])
            
            # Try to find question patterns
            print("\n  --- Looking for question patterns ---")
            import re
            
            # Find all potential question starts
            patterns = [
                (r'(?:^|\n)\s*(\d{1,2})\s*[.)\]]\s*([^\n]{20,100})', 'Pattern 1: "1. Question"'),
                (r'(?:^|\n)\s*[Qq]\.?\s*(\d{1,2})\s*[.)\]]?\s*([^\n]{20,100})', 'Pattern 2: "Q.1 Question"'),
                (r'(?:^|\n)\s*(\d{1,2})\)\s*([^\n]{20,100})', 'Pattern 3: "1) Question"'),
            ]
            
            for pattern, name in patterns:
                matches = re.findall(pattern, full_text, re.MULTILINE)
                if matches:
                    print(f"\n  {name}: Found {len(matches)} matches")
                    for i, match in enumerate(matches[:5]):  # Show first 5
                        print(f"    Q{match[0]}: {match[1][:60]}...")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_extraction()
