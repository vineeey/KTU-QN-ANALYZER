"""Views for analysis app."""
from django.views.generic import DetailView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.views import View
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
import re
import logging

logger = logging.getLogger(__name__)

from .models import AnalysisJob
from apps.papers.models import Paper
from apps.subjects.models import Subject, Module
from apps.questions.models import Question


# KTU 2019 Scheme Question to Module Mapping
KTU_MODULE_MAPPING = {
    # Part A (3 marks each)
    1: 1, 2: 1,   # Q1, Q2 -> Module 1
    3: 2, 4: 2,   # Q3, Q4 -> Module 2
    5: 3, 6: 3,   # Q5, Q6 -> Module 3
    7: 4, 8: 4,   # Q7, Q8 -> Module 4
    9: 5, 10: 5,  # Q9, Q10 -> Module 5
    # Part B (14 marks each)
    11: 1, 12: 1,  # Q11, Q12 -> Module 1
    13: 2, 14: 2,  # Q13, Q14 -> Module 2
    15: 3, 16: 3,  # Q15, Q16 -> Module 3
    17: 4, 18: 4,  # Q17, Q18 -> Module 4
    19: 5, 20: 5,  # Q19, Q20 -> Module 5
}


class AnalysisJobListView(LoginRequiredMixin, ListView):
    """List all analysis jobs for the user."""
    
    model = AnalysisJob
    template_name = 'analysis/job_list.html'
    context_object_name = 'jobs'
    paginate_by = 20
    
    def get_queryset(self):
        return AnalysisJob.objects.filter(
            paper__subject__user=self.request.user
        ).select_related('paper', 'paper__subject').order_by('-created_at')


class AnalysisStatusView(LoginRequiredMixin, View):
    """Get analysis job status (for HTMX polling)."""
    
    def get(self, request, pk):
        try:
            job = AnalysisJob.objects.get(
                pk=pk,
                paper__subject__user=request.user
            )
            return JsonResponse({
                'status': job.status,
                'progress': job.progress,
                'status_detail': job.status_detail,
                'questions_extracted': job.questions_extracted,
                'questions_classified': job.questions_classified,
                'duplicates_found': job.duplicates_found,
                'error_message': job.error_message,
            })
        except AnalysisJob.DoesNotExist:
            return JsonResponse({'error': 'Job not found'}, status=404)


class AnalysisDetailView(LoginRequiredMixin, DetailView):
    """View analysis job details."""
    
    model = AnalysisJob
    template_name = 'analysis/analysis_detail.html'
    context_object_name = 'job'
    
    def get_queryset(self):
        return AnalysisJob.objects.filter(paper__subject__user=self.request.user)


class ManualAnalyzeView(LoginRequiredMixin, View):
    """Manually trigger paper analysis (synchronous)."""
    
    def post(self, request, subject_pk):
        subject = get_object_or_404(Subject, pk=subject_pk, user=request.user)
        
        # Get pending papers
        pending_papers = subject.papers.filter(status='pending')
        
        if not pending_papers.exists():
            messages.info(request, 'No papers pending analysis.')
            return redirect('subjects:detail', pk=subject_pk)
        
        # Ensure modules exist (create 5 modules for KTU)
        if subject.modules.count() == 0:
            for i in range(1, 6):
                Module.objects.create(
                    subject=subject,
                    name=f'Module {i}',
                    number=i,
                    weightage=20
                )
        
        processed = 0
        failed = 0
        total_questions = 0
        errors = []
        
        for paper in pending_papers:
            try:
                paper.status = Paper.ProcessingStatus.PROCESSING
                paper.save()
                
                # Run KTU-specific analysis
                questions_count = self._analyze_ktu_paper(paper, subject)
                total_questions += questions_count
                processed += 1
                
            except Exception as e:
                paper.status = Paper.ProcessingStatus.FAILED
                paper.processing_error = str(e)
                paper.save()
                failed += 1
                errors.append(str(e))
                import traceback
                traceback.print_exc()
        
        # Run topic clustering after all papers are processed
        if processed > 0:
            try:
                from apps.analytics.clustering import analyze_subject_topics
                cluster_stats = analyze_subject_topics(subject, similarity_threshold=0.3)
                messages.success(
                    request, 
                    f'✅ Analyzed {processed} paper(s). Extracted {total_questions} questions. '
                    f'Created {cluster_stats["clusters_created"]} topic clusters.'
                )
            except Exception as e:
                messages.success(request, f'✅ Analyzed {processed} paper(s). Extracted {total_questions} questions.')
                messages.warning(request, f'⚠️ Topic clustering issue: {str(e)}')
        
        if failed > 0:
            messages.error(request, f'❌ {failed} paper(s) failed: {"; ".join(errors[:2])}')
        
        return redirect('subjects:detail', pk=subject_pk)
    
    def _analyze_ktu_paper(self, paper, subject):
        """Analyze a KTU format paper and extract questions."""
        from django.utils import timezone
        
        # Extract text from PDF using PyMuPDF
        text = self._extract_pdf_text(paper.file.path)
        
        if not text or len(text) < 100:
            raise Exception("Could not extract text from PDF. The file may be scanned/image-based.")
        
        paper.raw_text = text
        paper.save()
        
        # Parse exam info (month, year) from filename or text
        exam_info = self._parse_exam_info(paper.title, text)
        if exam_info.get('year'):
            paper.year = exam_info['year']
            paper.save()
        
        # Extract questions using improved KTU format parsing
        questions_data = self._extract_ktu_questions_improved(text)
        
        if not questions_data:
            raise Exception(f"No questions found in PDF. Text length: {len(text)} chars")
        
        # Get modules dictionary
        modules = {m.number: m for m in subject.modules.all()}
        
        # Create questions with module classification
        created_count = 0
        for q_data in questions_data:
            q_num = q_data['question_number']
            
            # Determine module using KTU mapping
            try:
                q_int = int(q_num)
                module_num = KTU_MODULE_MAPPING.get(q_int)
                module = modules.get(module_num) if module_num else None
                
                # Determine part and marks
                if q_int <= 10:
                    part = 'A'
                    marks = 3
                else:
                    part = 'B'
                    marks = 14
            except (ValueError, TypeError):
                module = None
                part = ''
                marks = q_data.get('marks')
            
            # Create question
            Question.objects.create(
                paper=paper,
                question_number=str(q_num),
                text=q_data['text'],
                marks=q_data.get('marks') or marks,
                part=part,
                module=module
            )
            created_count += 1
        
        # Mark paper as completed
        paper.status = Paper.ProcessingStatus.COMPLETED
        paper.processed_at = timezone.now()
        paper.save()
        
        return created_count
    
    def _extract_pdf_text(self, file_path):
        """Extract text from PDF using multiple fallback methods including OCR."""
        logger.info(f"Starting PDF extraction for: {file_path}")
        
        # Try PyPDF2 first (most reliable on Windows)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip() and len(text) > 100:
                logger.info(f"✅ PyPDF2 extraction successful: {len(text)} chars")
                return text
            elif text.strip():
                # Got some text but not much - might be scanned with minimal metadata
                logger.warning(f"⚠️ PyPDF2 extracted minimal text ({len(text)} chars), trying OCR")
        except Exception as e:
            logger.warning(f"❌ PyPDF2 extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip() and len(text) > 100:
                logger.info(f"✅ pdfplumber extraction successful: {len(text)} chars")
                return text
            elif text.strip():
                logger.warning(f"⚠️ pdfplumber extracted minimal text ({len(text)} chars), trying OCR")
        except Exception as e:
            logger.warning(f"❌ pdfplumber extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Try OCR extraction for scanned/image-based PDFs
        logger.info("🔍 Attempting OCR extraction (PDF appears to be scanned/image-based)")
        try:
            text = self._extract_with_direct_ocr(file_path)
            if text.strip():
                logger.info(f"✅ OCR extraction successful: {len(text)} chars")
                return text
            else:
                logger.error("❌ OCR returned empty text")
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        raise Exception("Could not extract text from PDF using any available library. The file may be scanned/image-based or corrupted.")
    
    def _extract_with_direct_ocr(self, file_path):
        """Extract text using direct PDF to image conversion and OCR."""
        import tempfile
        import subprocess
        from PIL import Image
        
        logger.info(f"🔍 Starting OCR extraction for: {file_path}")
        
        try:
            # Find Tesseract
            import shutil
            tesseract_path = shutil.which('tesseract')
            if not tesseract_path:
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Tesseract-OCR\tesseract.exe',
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        tesseract_path = path
                        break
            
            if not tesseract_path:
                raise Exception("Tesseract executable not found")
            
            logger.info(f"✅ Using Tesseract at: {tesseract_path}")
            
            # Try pdf2image if available
            try:
                from pdf2image import convert_from_path
                
                # Find poppler
                poppler_path = None
                possible_poppler = [
                    r'c:\poppler\poppler-24.08.0\Library\bin',
                    r'C:\Program Files\poppler\bin',
                    r'C:\poppler\bin',
                ]
                for path in possible_poppler:
                    if os.path.exists(path):
                        poppler_path = path
                        break
                
                logger.info(f"📄 Converting PDF to images (poppler: {poppler_path or 'system'})")
                
                # Convert PDF pages to images
                if poppler_path:
                    images = convert_from_path(file_path, poppler_path=poppler_path, dpi=300)
                else:
                    images = convert_from_path(file_path, dpi=300)
                
                logger.info(f"✅ Converted to {len(images)} page images")
                
                full_text = ""
                for i, image in enumerate(images):
                    logger.info(f"🔍 OCR processing page {i+1}/{len(images)}...")
                    text = self._ocr_image_subprocess(image, tesseract_path)
                    logger.info(f"   Page {i+1}: Extracted {len(text)} chars")
                    full_text += text + "\n"
                
                logger.info(f"✅ OCR complete. Total: {len(full_text)} chars")
                return full_text
                
            except ImportError as e:
                logger.error(f"❌ pdf2image not available: {e}")
                raise Exception(f"pdf2image required for OCR but not available: {e}")
            
        except Exception as e:
            logger.error(f"❌ Direct OCR failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _ocr_image_subprocess(self, image, tesseract_path):
        """Run Tesseract OCR using subprocess (fallback when pytesseract fails)."""
        import tempfile
        import subprocess
        
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                image.save(tmp_img.name, 'PNG')
                tmp_img_path = tmp_img.name
            
            # Run tesseract with stdout output (simpler and more reliable)
            cmd = [tesseract_path, tmp_img_path, 'stdout']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Get text from stdout
            text = result.stdout
            
            # Cleanup
            if os.path.exists(tmp_img_path):
                os.unlink(tmp_img_path)
            
            return text
            
        except Exception as e:
            logger.error(f"Subprocess OCR failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    
    def _parse_exam_info(self, title, text):
        """Parse exam info from title or text."""
        info = {'month': None, 'year': None}
        
        # Combined text to search
        search_text = f"{title} {text[:500]}"
        
        # Extract month
        month_match = re.search(
            r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|'
            r'JAN|FEB|MAR|APR|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',
            search_text, re.IGNORECASE
        )
        if month_match:
            info['month'] = month_match.group(1).title()
        
        # Extract year (2019-2025)
        year_match = re.search(r'(20[1-2][0-9])', search_text)
        if year_match:
            info['year'] = year_match.group(1)
        
        return info
    
    def _extract_ktu_questions_improved(self, text):
        """
        Extract questions from KTU format where:
        - PART A: Questions listed as plain lines (no numbers), marks column at end
        - PART B: Module sections with question numbers (11-20) and a) b) sub-parts
        """
        questions = []
        
        # ===== PART A EXTRACTION =====
        part_a_match = re.search(r'PART\s*A\s*\n(.+?)(?=PART\s*[B8]|$)', text, re.DOTALL | re.IGNORECASE)
        if part_a_match:
            part_a_text = part_a_match.group(1)
            
            # Remove header lines
            part_a_text = re.sub(r'\(Answer.*?\)', '', part_a_text, flags=re.IGNORECASE | re.DOTALL)
            part_a_text = re.sub(r'Marks?\s*$', '', part_a_text, flags=re.MULTILINE)
            
            # Split into lines and filter
            lines = [l.strip() for l in part_a_text.split('\n') if l.strip()]
            
            # Remove lines that are just numbers or very short
            question_lines = []
            for line in lines:
                # Skip header/footer/marks lines
                if re.match(r'^\d+\s*$', line):  # Just a number
                    continue
                if len(line) < 15:  # Too short
                    continue
                if re.match(r'^(Page|Marks|Duration|Course)', line, re.IGNORECASE):
                    continue
                # Remove trailing marks (single digit at end)
                line = re.sub(r'\s+\d\s*$', '', line).strip()
                if line:
                    question_lines.append(line)
            
            # Take first 10 lines as questions
            for i, q_text in enumerate(question_lines[:10], 1):
                questions.append({
                    'question_number': str(i),
                    'text': q_text[:2000],
                    'marks': 3
                })
                logger.debug(f"Part A Q{i}: {q_text[:60]}...")
        
        # ===== PART B EXTRACTION =====
        part_b_match = re.search(r'PART\s*[B8]\s*\n(.+?)(?=$)', text, re.DOTALL | re.IGNORECASE)
        if part_b_match:
            part_b_text = part_b_match.group(1)
            
            # TRY FORMAT 1 FIRST: Questions with numbers like "11 a)" or "I I a)" or "l1a)" (no space)
            q_pattern = r'(?:^|\n)\s*([Il1]\s*[Il1]|[Il1]?[12][0-9])\s*a\)'
            q_matches = list(re.finditer(q_pattern, part_b_text, re.MULTILINE))
            
            if len(q_matches) >= 7:  # Found most questions with Format 1 (lowered threshold)
                q_num = 11
                for i, match in enumerate(q_matches):
                    q_num_raw = match.group(1)
                    # Convert OCR errors: l1 -> 11, I1 -> 11, "I  I" -> 11
                    q_num_raw = q_num_raw.replace('l', '1').replace('I', '1').replace(' ', '')
                    try:
                        q_num = int(q_num_raw)
                    except (ValueError, TypeError):
                        continue
                    
                    if not (11 <= q_num <= 20):
                        continue
                    
                    # Extract text until next question
                    start_pos = match.start()
                    if i + 1 < len(q_matches):
                        end_pos = q_matches[i + 1].start()
                    else:
                        end_pos = len(part_b_text)
                    
                    q_section = part_b_text[start_pos:end_pos]
                    
                    # Extract a) and b) parts
                    parts = []
                    a_match = re.search(r'a\)\s*(.+?)(?=\s*b\)|$)', q_section, re.DOTALL | re.IGNORECASE)
                    if a_match:
                        a_text = a_match.group(1).strip()
                        a_text = re.sub(r'\s+\d+\s*$', '', a_text).strip()
                        a_text = ' '.join(a_text.split())
                        if len(a_text) > 10:
                            parts.append(a_text)
                    
                    b_match = re.search(r'b\)\s*(.+?)(?=$)', q_section, re.DOTALL | re.IGNORECASE)
                    if b_match:
                        b_text = b_match.group(1).strip()
                        b_text = re.sub(r'\s+\d+\s*$', '', b_text).strip()
                        b_text = ' '.join(b_text.split())
                        if len(b_text) > 10:
                            parts.append(b_text)
                    
                    if parts:
                        question_text = ' OR '.join(parts)
                        questions.append({
                            'question_number': str(q_num),
                            'text': question_text[:2000],
                            'marks': 14
                        })
                        logger.debug(f"Part B Q{q_num}: {question_text[:60]}...")
            
            else:
                # FORMAT 2: Module-based layout (DEC 2022 style)
                # Split by Module sections (Module -1, Module -2, etc.)
                module_pattern = r'Module\s*-\s*([1-5])'
                module_splits = re.split(module_pattern, part_b_text, flags=re.IGNORECASE)
                
                q_num = 11
                
                # Process each module section (module_splits has: [before, '1', content1, '2', content2, ...])
                for i in range(1, len(module_splits), 2):
                    if i + 1 >= len(module_splits):
                        break
                        
                    module_num = module_splits[i]
                    module_content = module_splits[i + 1]
                    
                    # Find all a) and b) parts in this module
                    # Pattern: a) text until b) or next a) or Module
                    ab_pattern = r'(a\)|b\))\s*([^\n]+(?:\n(?!\s*(?:a\)|b\)|Module|Page|Marks|\d{4}|OR\s+a\)|OR\s+b\)))[^\n]+)*)'
                    ab_matches = list(re.finditer(ab_pattern, module_content, re.MULTILINE))
                    
                    # Group consecutive a) and b) pairs into questions
                    i_ab = 0
                    while i_ab < len(ab_matches) and q_num <= 20:
                        match = ab_matches[i_ab]
                        part_type = match.group(1)
                        
                        if part_type == 'a)':
                            # Extract a) text
                            a_text = match.group(2).strip()
                            a_text = re.sub(r'\s+\d+\s*$', '', a_text).strip()
                            a_text = ' '.join(a_text.split())
                            
                            # Look for b) part
                            b_text = ""
                            if i_ab + 1 < len(ab_matches):
                                next_match = ab_matches[i_ab + 1]
                                if next_match.group(1) == 'b)':
                                    b_text = next_match.group(2).strip()
                                    b_text = re.sub(r'\s+\d+\s*$', '', b_text).strip()
                                    b_text = ' '.join(b_text.split())
                                    i_ab += 2  # Skip both a) and b)
                                else:
                                    i_ab += 1  # Only skip a)
                            else:
                                i_ab += 1
                            
                            # Create question
                            if len(a_text) > 10:
                                combined = a_text
                                if len(b_text) > 10:
                                    combined += f" OR {b_text}"
                                
                                questions.append({
                                    'question_number': str(q_num),
                                    'text': combined[:2000],
                                    'marks': 14
                                })
                                logger.debug(f"Part B Q{q_num}: {combined[:60]}...")
                                q_num += 1
                        else:
                            i_ab += 1  # Skip orphaned b)
        
        # COUNT PART B QUESTIONS
        part_b_count = sum(1 for q in questions if int(q['question_number']) > 10)
        
        # If we still don't have enough Part B questions, try aggressive extraction
        if part_b_count < 8:
            logger.warning(f"Only {part_b_count} Part B questions found, trying aggressive extraction")
            # Try to find questions just by looking for any "11", "12", etc. anywhere in the text
            # This handles formats like: "11 a)" or "a)11" or "l1 a)" 
            part_b_match = re.search(r'PART\s*[B8]\s*\n(.+?)(?=$)', text, re.DOTALL | re.IGNORECASE)
            if part_b_match:
                part_b_text = part_b_match.group(1)
                
                # Look for "11 a)", "12 a)", etc. OR "a)11", "a)12", etc.
                aggressive_pattern = r'(?:([Il1]?[12][0-9])\s*a\)|a\)\s*([Il1]?[12][0-9]))'
                agg_matches = list(re.finditer(aggressive_pattern, part_b_text, re.IGNORECASE))
                
                for match in agg_matches:
                    q_num_raw = match.group(1) or match.group(2)
                    q_num_raw = q_num_raw.replace('l', '1').replace('I', '1')
                    try:
                        q_num = int(q_num_raw)
                    except (ValueError, TypeError):
                        continue
                    
                    if not (11 <= q_num <= 20):
                        continue
                    
                    # Check if we already have this question
                    if any(q['question_number'] == str(q_num) for q in questions):
                        continue
                    
                    # Extract text around this match
                    start_pos = match.start()
                    # Look for "b)" part
                    b_search = re.search(r'b\)', part_b_text[start_pos:start_pos+800])
                    
                    if b_search:
                        # Found b), extract until next question or 500 chars
                        end_search = re.search(r'(?:[Il1]?[12][0-9]\s*a\)|a\)\s*[Il1]?[12][0-9])', part_b_text[start_pos + b_search.end():start_pos + b_search.end() + 500])
                        if end_search:
                            end_pos = start_pos + b_search.end() + end_search.start()
                        else:
                            end_pos = start_pos + b_search.end() + 400
                    else:
                        # No b), take next 400 chars
                        end_pos = start_pos + 400
                    
                    full_text = part_b_text[start_pos:end_pos]
                    
                    # Extract a) and b) parts
                    a_match = re.search(r'a\)\s*(.+?)(?=\s*b\)|$)', full_text, re.DOTALL | re.IGNORECASE)
                    b_match = re.search(r'b\)\s*(.+?)(?=$)', full_text, re.DOTALL | re.IGNORECASE)
                    
                    parts = []
                    if a_match:
                        a_text = a_match.group(1).strip()
                        a_text = re.sub(r'^\s*[Il1]?[12][0-9]\s*', '', a_text)  # Remove question number
                        a_text = re.sub(r'\s+\d+\s*$', '', a_text).strip()
                        a_text = ' '.join(a_text.split())
                        if len(a_text) > 15:
                            parts.append(a_text)
                    
                    if b_match:
                        b_text = b_match.group(1).strip()
                        b_text = re.sub(r'\s+\d+\s*$', '', b_text).strip()
                        b_text = ' '.join(b_text.split())
                        if len(b_text) > 15:
                            parts.append(b_text)
                    
                    if parts:
                        combined = ' OR '.join(parts)
                        questions.append({
                            'question_number': str(q_num),
                            'text': combined[:2000],
                            'marks': 14
                        })
                        logger.debug(f"Part B Q{q_num} (aggressive): {combined[:60]}...")
        
        logger.info(f"Extracted {len(questions)} questions (Part A: {sum(1 for q in questions if int(q['question_number']) <= 10)}, Part B: {sum(1 for q in questions if int(q['question_number']) > 10)})")
        
        # Fallback if not enough
        if len(questions) < 15:
            logger.warning(f"Only {len(questions)} questions extracted, trying fallback")
            return self._regex_fallback_extraction(text, questions)
        
        return questions
    
    def _regex_fallback_extraction(self, text, existing_questions):
        """Fallback regex-based extraction."""
        questions = {q['question_number']: q for q in existing_questions}
        
        # Try multiple aggressive patterns
        patterns = [
            r'(?:^|\n)\s*(\d{1,2})\s*[.)\]]\s*([^\n]{15,})',  # Simple numbered lines
            r'(?:^|\n)\s*[Qq]\.?\s*(\d{1,2})\s*[.)\]]?\s*([^\n]{15,})',  # Q prefix
            r'(\d{1,2})\s*[.)\]]\s*([a-z]\).*?)(?=\d{1,2}\s*[.)\]]|\Z)',  # With sub-parts
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                q_num = match[0].strip()
                q_text = match[1].strip()
                
                try:
                    q_int = int(q_num)
                    if 1 <= q_int <= 20 and len(q_text) >= 15:
                        if q_num not in questions:
                            marks = 3 if q_int <= 10 else 14
                            questions[q_num] = {
                                'question_number': q_num,
                                'text': q_text[:2000],
                                'marks': marks
                            }
                except ValueError:
                    pass
        
        result = list(questions.values())
        result.sort(key=lambda x: int(x['question_number']))
        return result
    
    def _parse_questions_line_by_line(self, text, existing_questions):
        """Fallback: parse questions line by line."""
        lines = text.split('\n')
        current_q = None
        questions = existing_questions.copy()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a question number
            match = re.match(r'^(\d{1,2})\s*[.)\]:\s]+(.*)$', line)
            if match:
                q_num = match.group(1)
                q_text = match.group(2).strip()
                
                try:
                    q_int = int(q_num)
                    if 1 <= q_int <= 20:
                        # Save previous question
                        if current_q and current_q['question_number'] not in questions:
                            if len(current_q['text']) >= 10:
                                questions[current_q['question_number']] = current_q
                        
                        # Start new question
                        current_q = {
                            'question_number': q_num,
                            'text': q_text,
                            'marks': None
                        }
                        continue
                except ValueError:
                    pass
            
            # If we have a current question, append this line if it looks like continuation
            if current_q:
                # Skip header/footer lines
                skip_patterns = [
                    r'^(PART|MODULE|SECTION|REG|TIME|MAX|COURSE|CODE|SCHEME|SEMESTER|BRANCH)',
                    r'^Page\s+\d+',
                    r'^\d+\s*$',
                    r'^[A-Z]{2,3}\d{3}',
                ]
                should_skip = any(re.match(p, line, re.IGNORECASE) for p in skip_patterns)
                
                if not should_skip and len(line) > 3:
                    current_q['text'] += ' ' + line
        
        # Don't forget the last question
        if current_q and current_q['question_number'] not in questions:
            if len(current_q['text']) >= 10:
                questions[current_q['question_number']] = current_q
        
        return questions


class ResetAndAnalyzeView(LoginRequiredMixin, View):
    """Reset all papers to pending and re-run analysis."""
    
    def post(self, request, subject_pk):
        subject = get_object_or_404(Subject, pk=subject_pk, user=request.user)
        
        # Delete all existing questions for this subject
        Question.objects.filter(paper__subject=subject).delete()
        
        # Delete existing topic clusters
        from apps.analytics.models import TopicCluster
        TopicCluster.objects.filter(subject=subject).delete()
        
        # Reset all papers to pending
        papers = subject.papers.all()
        papers.update(status='pending', processing_error='')
        
        messages.info(request, f'Reset {papers.count()} paper(s). Click "Start Analysis" to re-analyze.')
        
        return redirect('subjects:detail', pk=subject_pk)
