"""Views for analysis app."""
from django.views.generic import DetailView
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
        """Extract text from PDF using multiple fallback methods."""
        # Try PyPDF2 first (most reliable on Windows)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Try PyMuPDF as second option
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Try pdfplumber as last resort
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        raise Exception("Could not extract text from PDF using any available library. The file may be scanned/image-based or corrupted.")
    
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
        Improved extraction for KTU exam papers.
        KTU format: Questions listed line by line with marks at end
        Part A: 10 questions (3 marks each)
        Part B: 10 questions (14 marks each, with a) b) sub-parts)
        """
        questions = []
        
        # Find PART A section
        part_a_match = re.search(r'PART\s*A\s*\n(.+?)(?=PART\s*B|Module|$)', text, re.DOTALL | re.IGNORECASE)
        if part_a_match:
            part_a_text = part_a_match.group(1)
            # Extract lines that end with marks (e.g., "3" or "3 marks")
            part_a_lines = part_a_text.split('\n')
            q_num = 1
            for line in part_a_lines:
                line = line.strip()
                # Look for lines ending with a number (marks)
                if line and len(line) > 10:
                    # Check if line ends with marks indication
                    marks_match = re.search(r'\s+(\d+)\s*$', line)
                    if marks_match:
                        marks = int(marks_match.group(1))
                        # Remove marks from question text
                        question_text = re.sub(r'\s+\d+\s*$', '', line).strip()
                        if question_text and q_num <= 10:
                            questions.append({
                                'question_number': str(q_num),
                                'text': question_text[:2000],
                                'marks': marks
                            })
                            logger.debug(f"Part A Q{q_num}: {question_text[:50]}...")
                            q_num += 1
        
        # Find PART B section
        part_b_match = re.search(r'PART\s*B\s*\n(.+?)(?=$)', text, re.DOTALL | re.IGNORECASE)
        if part_b_match:
            part_b_text = part_b_match.group(1)
            # Part B has module sections with questions labeled 11, 12, 13, etc.
            # Look for patterns like "11", "l1", "12", "l2", etc. (with OCR errors)
            
            # Find all module sections
            module_sections = re.split(r'Module\s*[-:]?\s*\d+', part_b_text, flags=re.IGNORECASE)
            
            q_num = 11
            for section in module_sections[1:]:  # Skip first empty split
                if q_num > 20:
                    break
                    
                # Look for a) and b) sub-questions with marks
                lines = section.split('\n')
                current_q_parts = []
                
                for line in lines:
                    line = line.strip()
                    # Check for a) or b) patterns
                    if re.match(r'^[ab]\)', line) or 'a)' in line.lower() or 'b)' in line.lower():
                        # Extract the sub-question text
                        sub_q = re.sub(r'^[ab]\)\s*', '', line, flags=re.IGNORECASE).strip()
                        if sub_q and len(sub_q) > 10:
                            # Remove marks if present
                            sub_q = re.sub(r'\s+\d+\s*$', '', sub_q).strip()
                            current_q_parts.append(sub_q)
                
                # If we found sub-parts, create a question
                if current_q_parts and q_num <= 20:
                    question_text = ' OR '.join(current_q_parts)
                    questions.append({
                        'question_number': str(q_num),
                        'text': question_text[:2000],
                        'marks': 14
                    })
                    logger.debug(f"Part B Q{q_num}: {question_text[:50]}...")
                    q_num += 1
        
        logger.info(f"Extracted {len(questions)} questions from KTU paper (Part A: {len([q for q in questions if int(q['question_number']) <= 10])}, Part B: {len([q for q in questions if int(q['question_number']) > 10])})")
        
        # If we didn't get enough questions, try the old method as fallback
        if len(questions) < 15:
            logger.warning(f"Only found {len(questions)} questions with new method, trying fallback")
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
