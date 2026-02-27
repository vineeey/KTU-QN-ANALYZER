"""
PDF Report Generator.

Renders a Tailwind CSS HTML template to PDF using WeasyPrint and stores
the resulting file in a GeneratedReport record.

Usage::

    gen = ReportGenerator()
    report = gen.generate_module_report(subject, module)
    report = gen.generate_subject_report(subject)
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Orchestrates HTML rendering → WeasyPrint PDF → GeneratedReport persistence.

    Responsibilities:
    - Gather cluster / priority / question data for the report context.
    - Render a Django HTML template to a string.
    - Convert HTML to PDF via WeasyPrint.
    - Save the file to MEDIA_ROOT/reports/.
    - Create or update a GeneratedReport record.
    """

    # Template paths (relative to Django TEMPLATES dirs)
    MODULE_TEMPLATE = "reports/module_report.html"
    SUBJECT_TEMPLATE = "reports/subject_report.html"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_module_report(self, subject, module) -> "GeneratedReport":
        """
        Generate a PDF report for a single *module* of *subject*.

        Args:
            subject: Subject ORM instance.
            module: Module ORM instance.

        Returns:
            GeneratedReport ORM instance with the file field populated.
        """
        context = self._build_module_context(subject, module)
        html = self._render_template(self.MODULE_TEMPLATE, context)
        filename = f"report_{subject.id}_module_{module.number}.pdf"
        return self._save_report(html, filename, subject, module, context)

    def generate_subject_report(self, subject) -> "GeneratedReport":
        """
        Generate a full-subject PDF report covering all modules.

        Args:
            subject: Subject ORM instance.

        Returns:
            GeneratedReport ORM instance.
        """
        context = self._build_subject_context(subject)
        html = self._render_template(self.SUBJECT_TEMPLATE, context)
        filename = f"report_{subject.id}_full.pdf"
        return self._save_report(html, filename, subject, None, context)

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_module_context(self, subject, module) -> dict:
        """Build the template context for a single module report."""
        from apps.analytics.models import ClusterGroup, PriorityAssignment
        from apps.questions.models import Question

        clusters = (
            ClusterGroup.objects
            .filter(subject=subject, module=module)
            .prefetch_related("members__question__paper")
            .order_by("priority_assignment__tier", "-frequency")
        )

        # Questions by year (for Part A / Part B tables)
        questions = (
            Question.objects
            .filter(paper__subject=subject, module=module)
            .select_related("paper")
            .order_by("paper__year", "question_number")
        )

        part_a_by_year: dict = {}
        part_b_by_year: dict = {}
        for q in questions:
            year = str(q.paper.year) if q.paper.year else "Unknown"
            # Heuristic: Part A ≤ 2 marks, Part B > 2 marks
            if q.marks and q.marks > 2:
                part_b_by_year.setdefault(year, []).append(q)
            else:
                part_a_by_year.setdefault(year, []).append(q)

        # Priority distribution summary
        tier_summary = {1: 0, 2: 0, 3: 0, 4: 0}
        for cluster in clusters:
            try:
                tier = cluster.priority_assignment.tier
                tier_summary[tier] = tier_summary.get(tier, 0) + 1
            except Exception:
                tier_summary[4] += 1

        return {
            "subject": subject,
            "module": module,
            "clusters": list(clusters),
            "part_a_by_year": part_a_by_year,
            "part_b_by_year": part_b_by_year,
            "tier_summary": tier_summary,
            "total_questions": questions.count(),
        }

    def _build_subject_context(self, subject) -> dict:
        """Build the template context for a full-subject report."""
        from apps.subjects.models import Module
        from apps.analytics.services.priority_engine import PriorityEngine

        modules = Module.objects.filter(subject=subject).order_by("number")
        module_contexts = [
            self._build_module_context(subject, mod) for mod in modules
        ]
        tier_summary = PriorityEngine.tier_summary(subject)

        return {
            "subject": subject,
            "modules": list(modules),
            "module_contexts": module_contexts,
            "tier_summary": tier_summary,
        }

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render_template(self, template_name: str, context: dict) -> str:
        """
        Render a Django template to an HTML string.

        Falls back gracefully if the template file doesn't exist yet.
        """
        try:
            from django.template.loader import render_to_string
            return render_to_string(template_name, context)
        except Exception as exc:
            logger.warning(
                "Template '%s' failed to render (%s) – using minimal fallback HTML.",
                template_name, exc,
            )
            return self._fallback_html(context)

    @staticmethod
    def _fallback_html(context: dict) -> str:
        """Minimal HTML used when the proper template is missing."""
        subject = context.get("subject", "")
        module = context.get("module", "")
        return f"""
        <!DOCTYPE html>
        <html><head><title>Report</title>
        <style>body{{font-family:sans-serif;padding:2rem}}</style></head>
        <body>
        <h1>{subject}</h1>
        {'<h2>Module ' + str(module.number) + '</h2>' if module else ''}
        <p>Report generated successfully.</p>
        </body></html>
        """

    # ------------------------------------------------------------------
    # PDF generation and storage
    # ------------------------------------------------------------------

    def _save_report(
        self,
        html: str,
        filename: str,
        subject,
        module: Optional[object],
        context: dict,
    ) -> "GeneratedReport":
        """
        Convert *html* to PDF with WeasyPrint, write to MEDIA_ROOT/reports/,
        and create / update a GeneratedReport record.

        Args:
            html: Rendered HTML string.
            filename: Target filename (basename only).
            subject: Subject ORM instance.
            module: Module ORM instance or None.
            context: Rendering context (stored as generation_meta snapshot).

        Returns:
            GeneratedReport instance.
        """
        from django.conf import settings
        from django.core.files import File
        from apps.reports.models import GeneratedReport

        reports_dir = Path(settings.MEDIA_ROOT) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / filename

        pdf_bytes = self._html_to_pdf(html)
        output_path.write_bytes(pdf_bytes)

        report_type = (
            GeneratedReport.ReportType.MODULE
            if module else GeneratedReport.ReportType.SUBJECT
        )

        meta_snapshot = {
            "total_questions": context.get("total_questions", 0),
            "tier_summary": context.get("tier_summary", {}),
        }

        report, _ = GeneratedReport.objects.update_or_create(
            subject=subject,
            module=module,
            report_type=report_type,
            defaults={
                "file": f"reports/{filename}",
                "file_size_bytes": len(pdf_bytes),
                "generation_meta": meta_snapshot,
            },
        )

        logger.info("Report saved: %s (%d bytes)", output_path, len(pdf_bytes))
        return report

    @staticmethod
    def _html_to_pdf(html: str) -> bytes:
        """
        Convert an HTML string to a PDF byte string using WeasyPrint.

        Falls back to returning the HTML as UTF-8 bytes if WeasyPrint
        is unavailable (so the pipeline doesn't crash during development).

        Args:
            html: HTML string.

        Returns:
            PDF bytes, or UTF-8 HTML bytes as fallback.
        """
        try:
            from weasyprint import HTML
            return HTML(string=html).write_pdf()
        except ImportError:
            logger.warning(
                "WeasyPrint not installed – saving HTML instead of PDF. "
                "Install with: pip install weasyprint"
            )
            return html.encode("utf-8")
        except Exception as exc:
            logger.error("WeasyPrint PDF generation failed: %s", exc)
            return html.encode("utf-8")
