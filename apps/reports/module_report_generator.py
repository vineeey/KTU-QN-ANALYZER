"""
Enhanced module-wise PDF report generator matching the expected format.
Generates separate PDFs for each module with:
- Part A questions grouped by year
- Part B questions grouped by year
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


class ModuleReportGenerator:
    """Generates detailed module-wise PDF reports."""
    
    def __init__(self, subject: Subject):
        self.subject = subject
    
    def generate_all_module_reports(self) -> Dict[int, Optional[str]]:
        """
        Generate PDF reports for all modules in the subject.
        
        Returns:
            Dictionary mapping module number to PDF file path
        """
        results = {}
        modules = self.subject.modules.all().order_by('number')
        
        for module in modules:
            pdf_path = self.generate_module_report(module)
            results[module.number] = pdf_path
        
        return results
    
    def generate_module_report(self, module: Module) -> Optional[str]:
        """
        Generate a PDF report for a single module.
        
        Args:
            module: Module instance
            
        Returns:
            Path to generated PDF file, or None on failure
        """
        try:
            from weasyprint import HTML, CSS
            
            # Gather all data for this module
            report_data = self._prepare_module_data(module)
            
            # Render HTML template
            html_content = render_to_string('reports/module_report_detailed.html', report_data)
            
            # Generate PDF
            output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(self.subject.id)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"Module_{module.number}_{self.subject.code or 'subject'}.pdf"
            output_path = output_dir / filename
            
            # Custom CSS for better formatting
            css_str = """
            @page { size: A4; margin: 2cm; }
            body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 11pt; line-height: 1.6; }
            """
            
            HTML(string=html_content).write_pdf(
                str(output_path),
                stylesheets=[CSS(string=css_str)]
            )
            
            logger.info(f"Generated module report: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.error("WeasyPrint not installed")
            return None
        except Exception as e:
            logger.error(f"Module report generation failed: {e}", exc_info=True)
            return None
    
    def _prepare_module_data(self, module: Module) -> Dict[str, Any]:
        """
        Prepare all data needed for the module report.
        """
        # Get all questions for this module
        questions = Question.objects.filter(
            module=module
        ).select_related('paper', 'topic_cluster').order_by('paper__year', 'question_number')

        part_a_questions = questions.filter(part='A')
        part_b_questions = questions.filter(part='B')

        # Group questions by part and year (display is still year-tagged inside a module)
        part_a_by_year = self._group_questions_by_year(part_a_questions)
        part_b_by_year = self._group_questions_by_year(part_b_questions)

        # Track which question numbers map to this module for display in headers
        part_a_numbers = sorted({q.question_number for q in part_a_questions if q.question_number})
        part_b_numbers = sorted({q.question_number for q in part_b_questions if q.question_number})

        years_analyzed = {
            q.paper.year
            for q in questions
            if getattr(q, "paper", None) and q.paper.year
        }
        total_years = len(years_analyzed)

        # Get topic clusters with priority tiers
        topic_clusters = TopicCluster.objects.filter(
            module=module
        ).order_by('-frequency_count', 'topic_name')

        # Group topics by priority tier using frequency thresholds
        topics_by_tier = self._group_topics_by_tier(topic_clusters, total_years)

        # Create study priority order (sorted list)
        study_priority = self._create_study_priority_order(topic_clusters, total_years)
        study_order_by_tier = self._group_priority_order_by_tier(study_priority)

        return {
            'subject': self.subject,
            'module': module,
            'part_a_by_year': part_a_by_year,
            'part_b_by_year': part_b_by_year,
            'part_a_numbers': part_a_numbers,
            'part_b_numbers': part_b_numbers,
            'topics_by_tier': topics_by_tier,
            'study_priority': study_priority,
            'study_order_by_tier': study_order_by_tier,
            'total_questions': questions.count(),
            'total_part_a': part_a_questions.count(),
            'total_part_b': part_b_questions.count(),
            'total_years': total_years,
        }
    
    def _group_questions_by_year(self, questions) -> Dict[str, List[Question]]:
        """Group questions by exam year."""
        grouped = defaultdict(list)
        for q in questions:
            year_label = q.paper.year or 'Unknown'
            grouped[year_label].append(q)
        return dict(sorted(grouped.items()))
    
    def _group_topics_by_tier(self, topic_clusters, total_years: int) -> Dict[str, List[Dict[str, Any]]]:
        """Group topic clusters by frequency-based priority tier."""
        grouped = defaultdict(list)

        for cluster in topic_clusters:
            freq = cluster.frequency_count or 0
            avg_marks = (cluster.total_marks / cluster.question_count) if cluster.question_count else 0
            confidence = round((freq / total_years) * 100, 1) if total_years else 0
            priority_score = round((2 * freq) + avg_marks, 1)
            if freq >= 4:
                tier_label = 'Top Priority'
            elif freq == 3:
                tier_label = 'High Priority'
            elif freq == 2:
                tier_label = 'Medium Priority'
            else:
                tier_label = 'Low Priority'

            grouped[tier_label].append({
                'topic_name': cluster.topic_name,
                'frequency': freq,
                'years_appeared': cluster.years_appeared,
                'representative_text': cluster.representative_text,
                'average_marks': round(avg_marks, 1),
                'priority_score': priority_score,
                'confidence': confidence,
                'part_a_count': cluster.part_a_count or 0,
                'part_b_count': cluster.part_b_count or 0,
            })

        tier_order = ['Top Priority', 'High Priority', 'Medium Priority', 'Low Priority']
        result = {}
        for tier in tier_order:
            result[tier] = sorted(
                grouped.get(tier, []),
                key=lambda item: (-item.get('priority_score', 0), -item.get('frequency', 0), item.get('topic_name', ''))
            )
        return result

    def _create_study_priority_order(self, topic_clusters, total_years: int) -> List[Dict[str, Any]]:
        """
        Create ordered list of topics for studying.
        Returns list of dicts with topic info and recommendations.
        """
        priority_order = []

        for cluster in topic_clusters:
            # Create study recommendation
            freq = cluster.frequency_count
            years = ', '.join(cluster.years_appeared) if cluster.years_appeared else 'N/A'
            avg_marks = (cluster.total_marks / cluster.question_count) if cluster.question_count else 0
            priority_score = round((2 * freq) + avg_marks, 1)
            confidence = round((freq / total_years) * 100, 1) if total_years else 0

            if freq >= 4:
                recommendation = f"Must learn first — appeared {freq} times."
            elif freq == 3:
                recommendation = f"Strongly prepare — appeared {freq} times."
            elif freq == 2:
                recommendation = f"Prepare next — appeared {freq} times."
            else:
                recommendation = f"Optional — appeared {freq} time."
            
            priority_order.append({
                'topic': cluster.topic_name,
                'frequency': freq,
                'years': years,
                'total_marks': cluster.total_marks,
                'tier': cluster.get_tier_label(),
                'recommendation': recommendation,
                'representative_text': cluster.representative_text[:200] if cluster.representative_text else '',
                'priority_score': priority_score,
                'confidence': confidence,
                'average_marks': round(avg_marks, 1),
                'part_a_count': cluster.part_a_count or 0,
                'part_b_count': cluster.part_b_count or 0,
            })

        priority_order.sort(
            key=lambda item: (-item.get('priority_score', 0), -item.get('frequency', 0), item.get('topic', ''))
        )
        for idx, item in enumerate(priority_order, 1):
            item['rank'] = idx

        return priority_order

    def _group_priority_order_by_tier(self, priority_order: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Bucket study priority items by tier label for final study order display."""
        grouped = {
            'tier_1': [],
            'tier_2': [],
            'tier_3': [],
            'tier_4': [],
        }

        for item in priority_order:
            tier = item.get('tier')
            if tier == 'Top Priority':
                grouped['tier_1'].append(item)
            elif tier == 'High Priority':
                grouped['tier_2'].append(item)
            elif tier == 'Medium Priority':
                grouped['tier_3'].append(item)
            else:
                grouped['tier_4'].append(item)

        return grouped


def generate_module_reports(subject: Subject) -> Dict[int, Optional[str]]:
    """
    Convenience function to generate all module reports for a subject.
    
    Args:
        subject: Subject instance
        
    Returns:
        Dictionary mapping module number to PDF file path
    """
    generator = ModuleReportGenerator(subject)
    return generator.generate_all_module_reports()
