import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from apps.reports.generator import ReportGenerator
from apps.reports.module_report_generator import ModuleReportGenerator
from apps.subjects.models import Subject

for s in Subject.objects.all():
    print(f"Generating reports for {s.id} | {s.name}")
    rg = ReportGenerator(s)
    mrg = ModuleReportGenerator(s)
    try:
        path1 = rg.generate_module_report()
        print(f" module summary: {path1}")
    except Exception as e:
        print(f" module summary failed: {e}")
    try:
        path2 = rg.generate_analytics_report()
        print(f" analytics: {path2}")
    except Exception as e:
        print(f" analytics failed: {e}")
    try:
        res = mrg.generate_all_module_reports()
        print(f" module detailed: {res}")
    except Exception as e:
        print(f" module detailed failed: {e}")
