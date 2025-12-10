"""Views for report generation and download."""
from django.views.generic import View, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404
from django.contrib import messages
from pathlib import Path
import zipfile
import tempfile

from apps.subjects.models import Subject, Module
from .generator import ReportGenerator
from .module_report_generator import ModuleReportGenerator
from .ktu_report_generator import KTUModuleReportGenerator


class ReportsListView(LoginRequiredMixin, TemplateView):
    """List available reports for a subject."""
    
    template_name = 'reports/reports_list_new.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        subject = get_object_or_404(
            Subject, pk=self.kwargs['subject_pk'], user=self.request.user
        )
        
        context['subject'] = subject
        context['modules'] = subject.modules.all().order_by('number')
        
        return context


class GenerateModuleReportView(LoginRequiredMixin, View):
    """Generate and download a specific module report using KTU format."""
    
    def get(self, request, subject_pk, module_number):
        subject = get_object_or_404(
            Subject, pk=subject_pk, user=request.user
        )
        module = get_object_or_404(Module, subject=subject, number=module_number)
        
        # Use KTU report generator for proper format
        generator = KTUModuleReportGenerator(subject)
        pdf_path = generator.generate_module_report(module)
        
        if pdf_path and Path(pdf_path).exists():
            filename = f"Module_{module.number}_{subject.code or subject.name.replace(' ', '_')}.pdf"
            return FileResponse(
                open(pdf_path, 'rb'),
                content_type='application/pdf',
                as_attachment=True,
                filename=filename
            )
        
        messages.error(request, f"Failed to generate report for Module {module.number}")
        raise Http404("Report generation failed")


class GenerateAllModuleReportsView(LoginRequiredMixin, View):
    """Generate reports for all modules and return as ZIP."""
    
    def get(self, request, subject_pk):
        subject = get_object_or_404(
            Subject, pk=subject_pk, user=request.user
        )
        
        # Use KTU report generator
        generator = KTUModuleReportGenerator(subject)
        results = generator.generate_all_module_reports()
        
        # Create a ZIP file with all module PDFs
        successful_pdfs = []
        for module_num, pdf_path in results.items():
            if pdf_path and Path(pdf_path).exists():
                successful_pdfs.append((module_num, pdf_path))
        
        if not successful_pdfs:
            messages.error(request, "No reports could be generated")
            raise Http404("Report generation failed")
        
        # If only one report, return it directly
        if len(successful_pdfs) == 1:
            module_num, pdf_path = successful_pdfs[0]
            return FileResponse(
                open(pdf_path, 'rb'),
                content_type='application/pdf',
                as_attachment=True,
                filename=f"Module_{module_num}_{subject.code or subject.name.replace(' ', '_')}.pdf"
            )
        
        # Create ZIP file with all reports
        zip_path = tempfile.mktemp(suffix='.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for module_num, pdf_path in successful_pdfs:
                filename = f"Module_{module_num}_{subject.code or subject.name.replace(' ', '_')}.pdf"
                zf.write(pdf_path, filename)
        
        return FileResponse(
            open(zip_path, 'rb'),
            content_type='application/zip',
            as_attachment=True,
            filename=f"{subject.code or subject.name.replace(' ', '_')}_All_Modules.zip"
        )


class GenerateAnalyticsReportView(LoginRequiredMixin, View):
    """Generate and download analytics summary report."""
    
    def get(self, request, subject_pk):
        subject = get_object_or_404(
            Subject, pk=subject_pk, user=request.user
        )
        
        generator = ReportGenerator(subject)
        pdf_path = generator.generate_analytics_report()
        
        if pdf_path and Path(pdf_path).exists():
            return FileResponse(
                open(pdf_path, 'rb'),
                content_type='application/pdf',
                as_attachment=True,
                filename=f'{subject.code or subject.name}_analytics_report.pdf'
            )
        
        messages.error(request, "Failed to generate analytics report")
        raise Http404("Report generation failed")
