"""
PDF report generator using WeasyPrint.
"""
import logging
from pathlib import Path
from typing import Optional
from django.template.loader import render_to_string
from django.conf import settings

from apps.subjects.models import Subject
from apps.analytics.calculator import StatsCalculator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates PDF reports for subjects."""
    
    def __init__(self, subject: Subject):
        self.subject = subject
        self.calculator = StatsCalculator(subject)
    
    def generate_module_report(self) -> Optional[str]:
        """
        Generate module-wise question report.
        
        Returns:
            Path to generated PDF file, or None on failure
        """
        try:
            from weasyprint import HTML
            
            # Gather data
            stats = self.calculator.get_complete_stats()
            modules = self.subject.modules.all().prefetch_related('questions')
            
            # Prepare module data with questions
            module_data = []
            for module in modules:
                questions = module.questions.select_related('paper').order_by('paper__year')
                module_data.append({
                    'module': module,
                    'questions': questions,
                    'count': questions.count(),
                })
            
            # Render HTML template
            html_content = render_to_string('reports/module_report.html', {
                'subject': self.subject,
                'stats': stats,
                'modules': module_data,
            })
            
            # Generate PDF
            output_dir = Path(settings.MEDIA_ROOT) / 'reports'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"module_report_{self.subject.id}.pdf"
            output_path = output_dir / filename
            
            HTML(string=html_content).write_pdf(str(output_path))
            
            return str(output_path)
            
        except ImportError:
            logger.error("WeasyPrint not installed")
            return None
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def generate_analytics_report(self) -> Optional[str]:
        """
        Generate analytics summary report.
        """
        try:
            from weasyprint import HTML
            
            stats = self.calculator.get_complete_stats()
            
            html_content = render_to_string('reports/analytics_report.html', {
                'subject': self.subject,
                'stats': stats,
            })
            
            output_dir = Path(settings.MEDIA_ROOT) / 'reports'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"analytics_report_{self.subject.id}.pdf"
            output_path = output_dir / filename
            
            HTML(string=html_content).write_pdf(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Analytics report generation failed: {e}")
            return None
