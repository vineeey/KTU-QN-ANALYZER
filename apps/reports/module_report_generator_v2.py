"""
Module-wise report generator following EXACT master prompt specification.
Generates PDFs with strict format compliance.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from django.conf import settings

logger = logging.getLogger(__name__)


class ModuleReportGenerator:
    """
    Generates module-wise exam-ready study PDFs.
    Follows EXACT format from master prompt - NO DEVIATIONS.
    """
    
    def __init__(self, subject, module_number: int):
        """
        Args:
            subject: Subject model instance
            module_number: Module number (1-5)
        """
        self.subject = subject
        self.module_number = module_number
        self.module_name = f"Module {module_number}"
    
    def generate(self, questions: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> str:
        """
        Generate complete module report PDF.
        
        Args:
            questions: All questions for this module
            clusters: Clustered/repeated questions with tiers
            
        Returns:
            Path to generated PDF
        """
        # Prepare data
        data = self._prepare_data(questions, clusters)
        
        # Generate PDF using WeasyPrint
        pdf_path = self._generate_pdf(data)
        
        logger.info(f"Generated module report: {pdf_path}")
        return pdf_path
    
    def _prepare_data(self, questions: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare structured data for template."""
        
        # Separate PART A and PART B
        part_a_questions = [q for q in questions if q.get('part') == 'A']
        part_b_questions = [q for q in questions if q.get('part') == 'B']
        
        # Group by year
        part_a_by_year = self._group_by_year(part_a_questions)
        part_b_by_year = self._group_by_year(part_b_questions)
        
        # Get question number ranges
        part_a_qn_range = self._get_question_number_range('A', self.module_number)
        part_b_qn_range = self._get_question_number_range('B', self.module_number)
        
        # Organize clusters by tier
        clusters_by_tier = self._organize_by_tier(clusters)
        
        # Get total years analyzed
        all_years = sorted(set(q.get('year') for q in questions if 'year' in q))
        
        return {
            'module_number': self.module_number,
            'module_name': self.module_name,
            'subject_name': self.subject.name if hasattr(self.subject, 'name') else 'Subject',
            'part_a_qn_range': part_a_qn_range,
            'part_b_qn_range': part_b_qn_range,
            'part_a_by_year': part_a_by_year,
            'part_b_by_year': part_b_by_year,
            'clusters_by_tier': clusters_by_tier,
            'all_years': all_years,
            'total_years': len(all_years)
        }
    
    def _group_by_year(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group questions by year and session."""
        by_year = defaultdict(list)
        
        for q in questions:
            year = q.get('year')
            session = q.get('session', '')
            key = f"{session} {year}" if session else str(year)
            by_year[key].append(q)
        
        # Sort by year (most recent first)
        sorted_years = sorted(by_year.keys(), key=lambda x: int(x.split()[-1]), reverse=True)
        
        return {year: by_year[year] for year in sorted_years}
    
    def _get_question_number_range(self, part: str, module_num: int) -> str:
        """
        Get question number range for module.
        PART A: Qn 1-10 (2 per module)
        PART B: Qn 11-20 (2 per module)
        """
        if part == 'A':
            start = (module_num - 1) * 2 + 1
            end = start + 1
        else:  # PART B
            start = 10 + (module_num - 1) * 2 + 1
            end = start + 1
        
        if start == end:
            return f"Qn {start}"
        else:
            return f"Qn {start}-{end}"
    
    def _organize_by_tier(self, clusters: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Organize clusters by priority tier."""
        by_tier = {1: [], 2: [], 3: [], 4: []}
        
        for cluster in clusters:
            tier = cluster.get('tier', 4)
            by_tier[tier].append(cluster)
        
        return by_tier
    
    def _generate_pdf(self, data: Dict[str, Any]) -> str:
        """Generate PDF using WeasyPrint and Jinja2."""
        from jinja2 import Environment, FileSystemLoader
        from weasyprint import HTML
        
        # Setup Jinja2
        template_dir = Path(__file__).parent.parent.parent / 'templates' / 'reports'
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Load template
        template = env.get_template('module_report.html')
        
        # Render HTML
        html_content = template.render(**data)
        
        # Output path
        output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(self.subject.id if hasattr(self.subject, 'id') else 'unknown')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Module_{self.module_number}.pdf"
        
        # Generate PDF
        HTML(string=html_content).write_pdf(str(output_path))
        
        return str(output_path)
    
    def _get_tier_label(self, tier: int) -> str:
        """Get tier label for display."""
        labels = {
            1: 'TOP PRIORITY — Repeated 4–6 Times',
            2: 'HIGH PRIORITY — Repeated 3 Times',
            3: 'MEDIUM PRIORITY — Repeated 2 Times',
            4: 'LOW PRIORITY — Appears Only Once'
        }
        return labels.get(tier, 'UNKNOWN')
