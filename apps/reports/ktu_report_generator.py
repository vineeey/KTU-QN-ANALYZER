"""
KTU Module-wise PDF Report Generator
Generates PDFs matching the exact expected format with:
- Module Title with subject name and scheme
- PART A questions grouped by year (3 marks each)
- PART B questions grouped by year with Qn numbers (14 marks each)
- Repeated Question Analysis with priority tiers
- Final Study Priority Order
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict
from django.template.loader import render_to_string
from django.conf import settings

from apps.subjects.models import Subject, Module
from apps.questions.models import Question
from apps.analytics.models import TopicCluster

logger = logging.getLogger(__name__)


class KTUModuleReportGenerator:
    """Generates KTU-style module-wise PDF reports."""
    
    def __init__(self, subject: Subject):
        self.subject = subject
    
    def generate_all_module_reports(self) -> Dict[int, Optional[str]]:
        """Generate PDF reports for all modules."""
        results = {}
        modules = self.subject.modules.all().order_by('number')
        
        for module in modules:
            pdf_path = self.generate_module_report(module)
            results[module.number] = pdf_path
            logger.info(f"Generated report for Module {module.number}: {pdf_path}")
        
        return results
    
    def generate_module_report(self, module: Module) -> Optional[str]:
        """Generate a PDF report for a single module using ReportLab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            # Prepare data
            report_data = self._prepare_module_data(module)
            
            # Output path
            output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(self.subject.id)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"Module_{module.number}.pdf"
            output_path = output_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            story = self._build_pdf_content(report_data)
            
            # Generate PDF
            doc.build(story)
            
            logger.info(f"Generated: {output_path}")
            return str(output_path)
            
        except ImportError as e:
            logger.error(f"ReportLab not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return None
    
    def _build_pdf_content(self, data: Dict[str, Any]) -> List:
        """Build PDF content using ReportLab elements."""
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.units import cm
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='black',
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        title_text = f"Module {data['module'].number} – {data['subject'].name} (KTU {data['scheme_year']} Scheme)"
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Part A Section
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='black',
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("PART A (3 Marks each)", heading_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=12,
            alignment=TA_CENTER,
            fontStyle='italic'
        )
        story.append(Paragraph(
            f"(Qn {data['part_a_qn_start']} & {data['part_a_qn_end']} from all papers belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        # Part A Questions
        if data['part_a_by_year']:
            for year_group in data['part_a_by_year']:
                year_style = ParagraphStyle(
                    'YearHeading',
                    parent=styles['Normal'],
                    fontSize=11,
                    fontName='Helvetica-Bold',
                    spaceAfter=6,
                    spaceBefore=10
                )
                story.append(Paragraph(year_group['year'], year_style))
                
                for q in year_group['questions']:
                    q_text = f"• {q['text']} — ({q['year_short']}, {q['marks']} marks)"
                    q_style = ParagraphStyle(
                        'Question',
                        parent=styles['Normal'],
                        fontSize=10,
                        leftIndent=20,
                        spaceAfter=4
                    )
                    story.append(Paragraph(q_text, q_style))
        else:
            story.append(Paragraph("<i>No Part A questions found for this module.</i>", styles['Normal']))
        
        story.append(Spacer(1, 0.5*cm))
        
        # Part B Section
        story.append(Paragraph("PART B (14 Marks each)", heading_style))
        story.append(Paragraph(
            f"(Qn {data['part_b_qn_start']} & {data['part_b_qn_end']} belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        # Part B Questions
        if data['part_b_by_year']:
            for year_group in data['part_b_by_year']:
                story.append(Paragraph(year_group['year'], year_style))
                
                for qn in year_group['questions']:
                    qn_style = ParagraphStyle(
                        'QnNumber',
                        parent=styles['Normal'],
                        fontSize=10,
                        fontName='Helvetica-Bold',
                        spaceAfter=4,
                        spaceBefore=8
                    )
                    story.append(Paragraph(f"Qn {qn['number']}", qn_style))
                    
                    for part in qn['parts']:
                        part_text = f"• {part['text']} — ({part['year_short']}, {part['marks']} marks)"
                        story.append(Paragraph(part_text, q_style))
        else:
            story.append(Paragraph("<i>No Part B questions found for this module.</i>", styles['Normal']))
        
        # Page break before priority analysis
        story.append(PageBreak())
        
        # Priority Analysis
        priority_title_style = ParagraphStyle(
            'PriorityTitle',
            parent=styles['Heading1'],
            fontSize=14,
            textColor='black',
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph(
            f"✅ Module {data['module'].number} – Repeated Questions<br/>(Prioritized List)",
            priority_title_style
        ))
        story.append(Paragraph("<i>(Highest repeated = highest priority)</i>", subtitle_style))
        
        # Priority Tiers
        tier_style = ParagraphStyle(
            'TierHeading',
            parent=styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceAfter=8,
            spaceBefore=15,
            backColor='#f0f0f0',
            borderPadding=6
        )
        
        topic_name_style = ParagraphStyle(
            'TopicName',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica-Bold',
            spaceAfter=2,
            leftIndent=15
        )
        
        topic_years_style = ParagraphStyle(
            'TopicYears',
            parent=styles['Normal'],
            fontSize=9,
            textColor='#555555',
            spaceAfter=6,
            leftIndent=15
        )
        
        # Top Priority
        if data['priority_analysis']['top']:
            story.append(Paragraph("TOP PRIORITY — Repeated 5+ Times", tier_style))
            for item in data['priority_analysis']['top']:
                story.append(Paragraph(f"{item['rank']}. {item['topic']}", topic_name_style))
                story.append(Paragraph(f"Appears in: {item['years']}", topic_years_style))
        
        # High Priority
        if data['priority_analysis']['high']:
            story.append(Paragraph("HIGH PRIORITY — Repeated 3-4 Times", tier_style))
            for item in data['priority_analysis']['high']:
                story.append(Paragraph(f"{item['rank']}. {item['topic']}", topic_name_style))
                story.append(Paragraph(f"Appears in: {item['years']}", topic_years_style))
        
        # Medium Priority
        if data['priority_analysis']['medium']:
            story.append(Paragraph("MEDIUM PRIORITY — Repeated 2 Times", tier_style))
            for item in data['priority_analysis']['medium']:
                story.append(Paragraph(f"{item['rank']}. {item['topic']}", topic_name_style))
                story.append(Paragraph(f"Appears in: {item['years']}", topic_years_style))
        
        # Low Priority
        if data['priority_analysis']['low']:
            story.append(Paragraph("LOW PRIORITY — Appears Only Once", tier_style))
            for item in data['priority_analysis']['low']:
                story.append(Paragraph(f"{item['rank']}. {item['topic']}", topic_name_style))
                story.append(Paragraph(f"Appears in: {item['years']}", topic_years_style))
        
        # Final Study Order
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("FINAL PRIORITIZED STUDY ORDER", heading_style))
        story.append(Paragraph("<i>If you want to score high, study in THIS order:</i>", subtitle_style))
        
        study_tier_style = ParagraphStyle(
            'StudyTierTitle',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12
        )
        
        # Study Tiers
        if data['study_order']['tier_1']:
            story.append(Paragraph("Tier 1 (Most repeated — must learn first)", study_tier_style))
            for item in data['study_order']['tier_1']:
                story.append(Paragraph(f"• {item['topic']}", q_style))
        
        if data['study_order']['tier_2']:
            story.append(Paragraph("Tier 2 (Frequently repeated)", study_tier_style))
            for item in data['study_order']['tier_2']:
                story.append(Paragraph(f"• {item['topic']}", q_style))
        
        if data['study_order']['tier_3']:
            story.append(Paragraph("Tier 3 (Moderately repeated)", study_tier_style))
            for item in data['study_order']['tier_3']:
                story.append(Paragraph(f"• {item['topic']}", q_style))
        
        if data['study_order']['tier_4']:
            story.append(Paragraph("Tier 4 (One-time but possible)", study_tier_style))
            for item in data['study_order']['tier_4']:
                story.append(Paragraph(f"• {item['topic']}", q_style))
        
        return story
    
    def _prepare_module_data(self, module: Module) -> Dict[str, Any]:
        """Prepare all data for the module report."""
        
        # Get all questions for this module
        questions = Question.objects.filter(
            module=module
        ).select_related('paper').order_by('paper__year', 'question_number')
        
        # Group Part A questions by year
        part_a_by_year = self._group_part_a_by_year(questions.filter(part='A'))
        
        # Group Part B questions by year with question numbers
        part_b_by_year = self._group_part_b_by_year(questions.filter(part='B'))
        
        # Get topic clusters for priority analysis
        clusters = TopicCluster.objects.filter(
            module=module,
            subject=self.subject
        ).order_by('-frequency_count')
        
        # Group clusters by priority tier
        priority_analysis = self._group_by_priority(clusters)
        
        # Create final study order
        study_order = self._create_study_order(clusters)
        
        # Calculate correct question numbers for KTU pattern
        # Part A: Module 1 -> Q1,Q2; Module 2 -> Q3,Q4; etc.
        # Part B: Module 1 -> Q11,Q12; Module 2 -> Q13,Q14; etc.
        part_a_qn_start = (module.number - 1) * 2 + 1
        part_a_qn_end = part_a_qn_start + 1
        part_b_qn_start = 10 + (module.number - 1) * 2 + 1  # Q11,Q12 for Module 1
        part_b_qn_end = part_b_qn_start + 1
        
        return {
            'subject': self.subject,
            'module': module,
            'part_a_by_year': part_a_by_year,
            'part_b_by_year': part_b_by_year,
            'priority_analysis': priority_analysis,
            'study_order': study_order,
            'scheme_year': '2019',
            'part_a_qn_start': part_a_qn_start,
            'part_a_qn_end': part_a_qn_end,
            'part_b_qn_start': part_b_qn_start,
            'part_b_qn_end': part_b_qn_end,
        }
    
    def _group_part_a_by_year(self, questions) -> List[Dict]:
        """Group Part A questions by year."""
        by_year = defaultdict(list)
        
        for q in questions:
            year = self._format_year(q.paper)
            by_year[year].append({
                'text': q.text,
                'marks': q.marks or 3,
                'year_short': self._short_year(q.paper),
            })
        
        # Sort by year and convert to list
        result = []
        for year in sorted(by_year.keys(), key=self._year_sort_key):
            result.append({
                'year': year,
                'questions': by_year[year]
            })
        return result
    
    def _group_part_b_by_year(self, questions) -> List[Dict]:
        """Group Part B questions by year with question numbers."""
        by_year = defaultdict(lambda: defaultdict(list))
        
        for q in questions:
            year = self._format_year(q.paper)
            q_num = q.question_number
            by_year[year][q_num].append({
                'text': q.text,
                'marks': q.marks or 14,
                'year_short': self._short_year(q.paper),
            })
        
        # Convert to structured format
        result = []
        for year in sorted(by_year.keys(), key=self._year_sort_key):
            year_data = {
                'year': year,
                'questions': []
            }
            for q_num in sorted(by_year[year].keys(), key=lambda x: int(x) if x.isdigit() else 0):
                year_data['questions'].append({
                    'number': q_num,
                    'parts': by_year[year][q_num]
                })
            result.append(year_data)
        return result
    
    def _group_by_priority(self, clusters) -> Dict[str, List]:
        """Group topic clusters by priority tier."""
        priority = {
            'top': [],      # 4+ times
            'high': [],     # 3 times
            'medium': [],   # 2 times
            'low': [],      # 1 time
        }
        
        rank = 1
        for cluster in clusters:
            freq = cluster.frequency_count
            years = cluster.years_appeared or []
            
            item = {
                'rank': rank,
                'topic': cluster.topic_name,
                'frequency': freq,
                'years': ', '.join(str(y) for y in years) if years else 'N/A',
                'description': cluster.representative_text[:200] if cluster.representative_text else '',
            }
            
            if freq >= 5:
                priority['top'].append(item)  # Top Priority: 5+ times
            elif freq >= 3:
                priority['high'].append(item)  # High Priority: 3-4 times
            elif freq == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
            
            rank += 1
        
        return priority
    
    def _create_study_order(self, clusters) -> Dict[str, List]:
        """Create final study priority order by tiers."""
        tiers = {
            'tier_1': [],  # Most repeated - must learn first
            'tier_2': [],  # Frequently repeated
            'tier_3': [],  # Moderately repeated
            'tier_4': [],  # One-time but possible
        }
        
        for cluster in clusters:
            freq = cluster.frequency_count
            item = {
                'topic': cluster.topic_name,
                'frequency': freq,
            }
            
            if freq >= 5:
                tiers['tier_1'].append(item)  # Most repeated (5+)
            elif freq >= 3:
                tiers['tier_2'].append(item)  # Frequently repeated (3-4)
            elif freq == 2:
                tiers['tier_3'].append(item)
            else:
                tiers['tier_4'].append(item)
        
        return tiers
    
    def _format_year(self, paper) -> str:
        """Format paper year for display (e.g., 'December 2021')."""
        if not paper:
            return 'Unknown'
        
        year = paper.year or ''
        title = paper.title or ''
        
        # Try to extract month from title
        import re
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
        
        for month in months:
            if month.upper() in title.upper():
                return f"{month} {year}" if year else month
        
        # Check for short month names
        short_months = {'JAN': 'January', 'FEB': 'February', 'MAR': 'March', 
                        'APR': 'April', 'MAY': 'May', 'JUN': 'June',
                        'JUL': 'July', 'AUG': 'August', 'SEP': 'September',
                        'OCT': 'October', 'NOV': 'November', 'DEC': 'December'}
        
        for short, full in short_months.items():
            if short in title.upper():
                return f"{full} {year}" if year else full
        
        return year if year else 'Unknown'
    
    def _short_year(self, paper) -> str:
        """Get short year format (e.g., 'Dec 2021')."""
        if not paper:
            return ''
        
        year = paper.year or ''
        title = paper.title or ''
        
        import re
        short_months = {
            'JANUARY': 'Jan', 'FEBRUARY': 'Feb', 'MARCH': 'Mar',
            'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
            'JULY': 'Jul', 'AUGUST': 'Aug', 'SEPTEMBER': 'Sep',
            'OCTOBER': 'Oct', 'NOVEMBER': 'Nov', 'DECEMBER': 'Dec'
        }
        
        for full, short in short_months.items():
            if full in title.upper():
                return f"{short} {year}"
        
        return year
    
    def _year_sort_key(self, year_str: str):
        """Sort key for year strings."""
        import re
        # Extract year number
        match = re.search(r'(\d{4})', year_str)
        year_num = int(match.group(1)) if match else 0
        
        # Month order
        month_order = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month_num = 0
        for month, num in month_order.items():
            if month in year_str.lower():
                month_num = num
                break
        
        return (year_num, month_num)
    
    def _get_report_css(self) -> str:
        """Get CSS for PDF styling - xhtml2pdf compatible."""
        return """
        @page {
            size: a4;
            margin: 2cm;
        }
        body {
            font-family: Arial, Helvetica, sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #000;
        }
        h1 {
            font-size: 18pt;
            color: #000;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        h2 {
            font-size: 14pt;
            color: #000;
            margin-top: 20px;
            margin-bottom: 10px;
            padding-bottom: 3px;
            border-bottom: 2px solid #000;
            font-weight: bold;
        }
        h3 {
            font-size: 12pt;
            color: #000;
            margin-top: 12px;
            margin-bottom: 6px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 15px;
            font-style: italic;
        }
        .year-heading {
            font-weight: bold;
            color: #000;
            margin-top: 12px;
            margin-bottom: 6px;
            font-size: 11pt;
        }
        .question-item {
            margin-left: 20px;
            margin-bottom: 6px;
            text-indent: -10px;
            padding-left: 10px;
        }
        .marks-tag {
            color: #555;
            font-style: italic;
        }
        .qn-number {
            font-weight: bold;
            color: #000;
            margin-top: 8px;
            margin-bottom: 4px;
        }
        .priority-section {
            margin-top: 25px;
            page-break-before: always;
        }
        .priority-heading {
            font-size: 14pt;
            color: #000;
            text-align: center;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .tier-heading {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 8px;
            padding: 6px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
        }
        .tier-top {
            background-color: #ffcccc;
        }
        .tier-high {
            background-color: #ffe6cc;
        }
        .tier-medium {
            background-color: #ffffcc;
        }
        .tier-low {
            background-color: #cce6ff;
        }
        .topic-item {
            margin: 8px 0 8px 15px;
        }
        .topic-name {
            font-weight: bold;
        }
        .topic-years {
            color: #555;
            font-size: 9pt;
        }
        .topic-desc {
            color: #444;
            font-size: 9pt;
            font-style: italic;
            margin-left: 15px;
        }
        .study-order {
            margin-top: 25px;
        }
        .study-tier {
            margin: 12px 0;
        }
        .study-tier-title {
            font-weight: bold;
            margin-bottom: 4px;
        }
        ul {
            margin-left: 20px;
            margin-top: 5px;
        }
        li {
            margin-bottom: 4px;
        }
        """


def generate_ktu_module_reports(subject: Subject) -> Dict[int, Optional[str]]:
    """Convenience function to generate all KTU module reports."""
    generator = KTUModuleReportGenerator(subject)
    return generator.generate_all_module_reports()
