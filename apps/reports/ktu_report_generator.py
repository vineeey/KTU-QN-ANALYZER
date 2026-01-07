"""
KTU Module Report Generator - EXACT MATCH to Expected Output
Generates PDFs with proper formatting, colors, emojis, and structure.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict
from django.conf import settings
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

from apps.subjects.models import Subject, Module
from apps.questions.models import Question
from apps.analytics.models import TopicCluster

logger = logging.getLogger(__name__)


class KTUModuleReportGenerator:
    """Generates KTU-style module reports matching expected output format."""
    
    def __init__(self, subject: Subject):
        self.subject = subject
        self.diamond = "◆"  # Red diamond bullet for years
    
    def generate_all_module_reports(self) -> Dict[int, Optional[str]]:
        """Generate reports for all 5 modules."""
        results = {}
        for module_num in range(1, 6):
            module = Module.objects.filter(subject=self.subject, number=module_num).first()
            if module:
                pdf_path = self.generate_module_report(module)
                results[module_num] = pdf_path
        return results
    
    def generate_module_report(self, module: Module) -> Optional[str]:
        """Generate single module PDF report."""
        try:
            # Output path
            output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(self.subject.id)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"Module_{module.number}.pdf"
            
            # Create PDF
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            data = self._prepare_data(module)
            story = self._build_content(data)
            
            # Generate
            doc.build(story)
            logger.info(f"Generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            return None
    
    def _prepare_data(self, module: Module) -> Dict[str, Any]:
        """Prepare all data for the report."""
        # Get all questions for this module
        questions = Question.objects.filter(
            module=module,
            paper__subject=self.subject
        ).select_related('paper').order_by('paper__year', 'question_number')
        
        # Calculate question numbers
        part_a_start = (module.number - 1) * 2 + 1
        part_a_end = part_a_start + 1
        part_b_start = 10 + (module.number - 1) * 2 + 1
        part_b_end = part_b_start + 1
        
        # Group questions
        part_a = self._group_part_a(questions.filter(part='A'))
        part_b = self._group_part_b(questions.filter(part='B'))
        
        years_analyzed = {
            q.paper.year
            for q in questions
            if getattr(q, "paper", None) and q.paper.year
        }

        # Analyze priorities based on question frequency (without TopicCluster)
        priority = self._analyze_priorities_from_questions(
            questions, total_years=len(years_analyzed)
        )
        
        return {
            'module': module,
            'subject': self.subject,
            'part_a_start': part_a_start,
            'part_a_end': part_a_end,
            'part_b_start': part_b_start,
            'part_b_end': part_b_end,
            'part_a': part_a,
            'part_b': part_b,
            'priority': priority,
        }
    
    def _build_content(self, data: Dict) -> List:
        """Build PDF content."""
        styles = getSampleStyleSheet()
        story = []
        
        # === TITLE ===
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=black
        )
        story.append(Paragraph(
            f"Module {data['module'].number} – DISASTER MANAGEMENT (KTU 2019 Scheme)",
            title_style
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # === PART A ===
        heading_style = ParagraphStyle(
            'Heading',
            fontSize=14,
            fontName='Helvetica-Bold',
            spaceAfter=8,
            spaceBefore=10,
            textColor=black
        )
        story.append(Paragraph("PART A (3 Marks each)", heading_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            fontSize=9,
            textColor=HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=15
        )
        story.append(Paragraph(
            f"(Qn {data['part_a_start']} & {data['part_a_end']} from all papers belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        # Part A Questions
        year_style = ParagraphStyle(
            'YearHeading',
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12,
            textColor=HexColor('#CC0000')  # Red
        )
        
        question_style = ParagraphStyle(
            'Question',
            fontSize=10,
            leftIndent=15,
            spaceAfter=4
        )
        
        for year_data in data['part_a']:
            # Red diamond + year
            story.append(Paragraph(
                f'<font color="#CC0000">◆</font> <b>{year_data["year"]}</b>',
                year_style
            ))
            
            for q in year_data['questions']:
                story.append(Paragraph(
                    f"• {q['text']} — ({q['year_short']}, {q['marks']} marks)",
                    question_style
                ))
        
        story.append(Spacer(1, 0.5*cm))
        
        # === PART B ===
        story.append(Paragraph("PART B (14 Marks each)", heading_style))
        story.append(Paragraph(
            f"(Qn {data['part_b_start']} & {data['part_b_end']} belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        qn_number_style = ParagraphStyle(
            'QnNumber',
            fontSize=10,
            fontName='Helvetica-Bold',
            spaceBefore=10,
            spaceAfter=4
        )
        
        for year_data in data['part_b']:
            # Red diamond + year
            story.append(Paragraph(
                f'<font color="#CC0000">◆</font> <b>{year_data["year"]}</b>',
                year_style
            ))
            
            for qn in year_data['questions']:
                story.append(Paragraph(f"Qn {qn['number']}", qn_number_style))
                
                for part in qn['parts']:
                    # Split OR parts into separate bullets
                    import re
                    or_parts = re.split(r'\s+OR\s+', part['text'], flags=re.IGNORECASE)
                    
                    for or_part in or_parts:
                        or_part = or_part.strip()
                        if or_part:
                            story.append(Paragraph(
                                f"• {or_part} — ({part['year_short']}, {part['marks']} marks)",
                                question_style
                            ))
        
        # === PAGE BREAK ===
        story.append(PageBreak())
        
        # === REPEATED QUESTIONS ANALYSIS ===
        section_title_style = ParagraphStyle(
            'SectionTitle',
            fontSize=14,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=10,
            spaceBefore=15,
            textColor=black
        )
        
        story.append(Paragraph(
            f"■ Module {data['module'].number} – Repeated Questions<br/>(Prioritized List)",
            section_title_style
        ))
        story.append(Paragraph(
            "<i>(Highest repeated = highest priority)</i>",
            subtitle_style
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # Priority tiers
        tier_title_style = ParagraphStyle(
            'TierTitle',
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceBefore=15,
            spaceAfter=8,
            textColor=black
        )
        
        topic_style = ParagraphStyle(
            'Topic',
            fontSize=10,
            leftIndent=15,
            spaceAfter=3
        )
        
        years_style = ParagraphStyle(
            'Years',
            fontSize=9,
            leftIndent=15,
            spaceAfter=10,
            textColor=HexColor('#555555')
        )
        
        # TOP PRIORITY
        if data['priority']['top']:
            story.append(Paragraph(
                '🔥 <b>TOP PRIORITY — Repeated 4 Times</b>',
                tier_title_style
            ))
            for item in data['priority']['top']:
                story.append(Paragraph(
                    f"<b>{item['rank']}. {item['topic']}</b>",
                    topic_style
                ))
                story.append(Paragraph(
                    f"Appears in: {item['years']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Priority Score: {item['priority_score']} • Confidence: {item['confidence']}% • Part A: {item['part_a']} • Part B: {item['part_b']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Avg Marks: {item['average_marks']} • Frequency (years): {item['frequency']}",
                    years_style
                ))
        
        # HIGH PRIORITY
        if data['priority']['high']:
            story.append(Paragraph(
                '🔥 <b>HIGH PRIORITY — Repeated 3 Times</b>',
                tier_title_style
            ))
            for item in data['priority']['high']:
                story.append(Paragraph(
                    f"<b>{item['rank']}. {item['topic']}</b>",
                    topic_style
                ))
                story.append(Paragraph(
                    f"Appears in: {item['years']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Priority Score: {item['priority_score']} • Confidence: {item['confidence']}% • Part A: {item['part_a']} • Part B: {item['part_b']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Avg Marks: {item['average_marks']} • Frequency (years): {item['frequency']}",
                    years_style
                ))
        
        # MEDIUM PRIORITY
        if data['priority']['medium']:
            story.append(Paragraph(
                '⚡ <b>MEDIUM PRIORITY — Repeated 2 Times</b>',
                tier_title_style
            ))
            for item in data['priority']['medium']:
                story.append(Paragraph(
                    f"<b>{item['rank']}. {item['topic']}</b>",
                    topic_style
                ))
                story.append(Paragraph(
                    f"Appears in: {item['years']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Priority Score: {item['priority_score']} • Confidence: {item['confidence']}% • Part A: {item['part_a']} • Part B: {item['part_b']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Avg Marks: {item['average_marks']} • Frequency (years): {item['frequency']}",
                    years_style
                ))
        
        # LOW PRIORITY
        if data['priority']['low']:
            story.append(Paragraph(
                '✓ <b>LOW PRIORITY — Appears Only Once (But still Module {data["module"].number})</b>',
                tier_title_style
            ))
            for item in data['priority']['low']:
                story.append(Paragraph(
                    f"<b>{item['rank']}. {item['topic']}</b>",
                    topic_style
                ))
                story.append(Paragraph(
                    f"Appears in: {item['years']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Priority Score: {item['priority_score']} • Confidence: {item['confidence']}% • Part A: {item['part_a']} • Part B: {item['part_b']}",
                    years_style
                ))
                story.append(Paragraph(
                    f"Avg Marks: {item['average_marks']} • Frequency (years): {item['frequency']}",
                    years_style
                ))
        
        # === FINAL PRIORITIZED STUDY ORDER ===
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            '📌 <b>FINAL PRIORITIZED STUDY ORDER</b><br/>(My recommended ranking)',
            section_title_style
        ))
        story.append(Paragraph(
            "<i>If you want to score high, study in THIS order:</i>",
            subtitle_style
        ))
        story.append(Spacer(1, 0.3*cm))
        
        tier_heading_style = ParagraphStyle(
            'TierHeading',
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6
        )
        
        list_style = ParagraphStyle(
            'ListItem',
            fontSize=10,
            leftIndent=20,
            spaceAfter=3
        )
        
        # Study order tiers
        if data['priority']['top']:
            story.append(Paragraph(
                "<b>Tier 1 (Most repeated — must learn first)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['top'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['high']:
            story.append(Paragraph(
                "<b>Tier 2 (Frequently repeated)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['high'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['medium']:
            story.append(Paragraph(
                "<b>Tier 3 (Moderately repeated)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['medium'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['low']:
            story.append(Paragraph(
                "<b>Tier 4 (One-time but possible)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['low'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        return story
    
    def _group_part_a(self, questions) -> List[Dict]:
        """Group Part A questions by year."""
        by_year = defaultdict(list)
        
        for q in questions:
            year = self._format_year(q.paper)
            by_year[year].append({
                'text': self._clean_text(q.text),
                'marks': q.marks or 3,
                'year_short': self._short_year(q.paper)
            })
        
        result = []
        for year in self._sorted_years(by_year.keys()):
            result.append({
                'year': year,
                'questions': by_year[year]
            })
        return result
    
    def _group_part_b(self, questions) -> List[Dict]:
        """Group Part B questions by year and question number."""
        by_year = defaultdict(lambda: defaultdict(list))
        
        for q in questions:
            year = self._format_year(q.paper)
            qn = q.question_number
            
            by_year[year][qn].append({
                'text': self._clean_text(q.text),
                'marks': q.marks or 14,
                'year_short': self._short_year(q.paper)
            })
        
        result = []
        for year in self._sorted_years(by_year.keys()):
            questions_list = []
            for qn in sorted(by_year[year].keys()):
                questions_list.append({
                    'number': qn,
                    'parts': by_year[year][qn]
                })
            result.append({
                'year': year,
                'questions': questions_list
            })
        return result
    
    def _analyze_priorities(self, clusters) -> Dict[str, List]:
        """Analyze topic clusters and create priority tiers."""
        priority = {
            'top': [],     # 4+ times
            'high': [],    # 3 times
            'medium': [],  # 2 times
            'low': []      # 1 time
        }
        
        rank = 1
        for cluster in clusters:
            freq = cluster.frequency_count
            years = cluster.years_appeared or []
            
            item = {
                'rank': rank,
                'topic': cluster.topic_name,
                'frequency': freq,
                'years': ', '.join(str(y) for y in years) if years else 'N/A'
            }
            
            if freq >= 4:
                priority['top'].append(item)
            elif freq == 3:
                priority['high'].append(item)
            elif freq == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
            
            rank += 1
        
        return priority
    
    def _analyze_priorities_from_questions(
        self, questions, total_years: int = 0
    ) -> Dict[str, List]:
        """Analyze priority topics directly from questions (fallback when clusters unavailable)."""
        from collections import Counter
        
        priority = {
            'top': [],     # Tier 1: 4+ times
            'high': [],    # Tier 2: 3 times
            'medium': [],  # Tier 3: 2 times
            'low': []      # Tier 4: 1 time
        }
        
        # Count question text occurrences with part-wise split
        question_stats = defaultdict(lambda: {
            "years": set(),
            "marks_total": 0,
            "marks_count": 0,
            "part_a": 0,
            "part_b": 0,
        })
        all_years = set()
        
        for q in questions:
            text = q.text[:100]  # Use first 100 chars as identifier
            stats = question_stats[text]
            if q.paper and q.paper.year:
                stats["years"].add(q.paper.year)
                all_years.add(q.paper.year)
            if q.marks:
                stats["marks_total"] += q.marks
                stats["marks_count"] += 1
            part = getattr(q, "part", None)
            if part == 'A':
                stats["part_a"] += 1
            elif part == 'B':
                stats["part_b"] += 1
        
        if not total_years:
            total_years = len(all_years)
        
        # Build scored topic list
        topics_scored = []
        for text, stats in question_stats.items():
            years = sorted(stats["years"])
            frequency_years = len(years)
            average_marks = (
                stats["marks_total"] / stats["marks_count"]
                if stats["marks_count"]
                else 0
            )
            priority_score = round((2 * frequency_years) + average_marks, 1)
            confidence = (
                round((frequency_years / total_years) * 100, 1)
                if total_years
                else 0
            )
            
            topics_scored.append({
                'topic': text[:80] + '...' if len(text) > 80 else text,
                'frequency': frequency_years,
                'years': ', '.join(str(y) for y in years) if years else 'N/A',
                'priority_score': priority_score,
                'average_marks': round(average_marks, 1),
                'confidence': confidence,
                'part_a': stats["part_a"],
                'part_b': stats["part_b"],
            })
        
        # Sort by priority score then frequency
        topics_scored.sort(
            key=lambda item: (-item['priority_score'], -item['frequency'], item['topic'])
        )
        
        # Bucket by tier using frequency across distinct years
        for rank, item in enumerate(topics_scored, 1):
            item['rank'] = rank
            freq = item['frequency']
            
            if freq >= 4:
                priority['top'].append(item)
            elif freq == 3:
                priority['high'].append(item)
            elif freq == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
        
        return priority
    
    def _format_year(self, paper) -> str:
        """Format year as 'December 2021'."""
        if paper and paper.exam_type:
            return paper.exam_type
        return 'Unknown'
    
    def _short_year(self, paper) -> str:
        """Format year as 'Dec 2021'."""
        if not paper or not paper.exam_type:
            return ''
        
        months = {
            'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
            'April': 'Apr', 'May': 'May', 'June': 'Jun',
            'July': 'Jul', 'August': 'Aug', 'September': 'Sep',
            'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
        }
        
        year_str = paper.exam_type
        for full, short in months.items():
            if full in year_str:
                return year_str.replace(full, short)
        return year_str
    
    def _clean_text(self, text: str) -> str:
        """Clean question text."""
        import re
        text = text.strip()
        # Remove leading numbers/formatting
        text = re.sub(r'^[Il1O0-9\.\)\s]+', '', text).strip()
        return text
    
    def _sorted_years(self, years) -> List[str]:
        """Sort years chronologically."""
        year_order = {
            'December 2021': 1,
            'December 2022': 2,
            'December 2023': 3,
            'June 2024': 4,
            'November 2024': 5,
            'May 2025': 6
        }
        return sorted(years, key=lambda y: year_order.get(y, 999))


def generate_ktu_module_reports(subject: Subject) -> Dict[int, Optional[str]]:
    """Generate all module reports."""
    generator = KTUModuleReportGenerator(subject)
    return generator.generate_all_module_reports()
