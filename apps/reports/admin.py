"""
Admin configuration for reports app.
"""
from django.contrib import admin
from .models import GeneratedReport


@admin.register(GeneratedReport)
class GeneratedReportAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'report_type', 'file_size_bytes', 'page_count', 'created_at')
    list_filter = ('report_type', 'subject')
    search_fields = ('subject__name', 'module__name')
    readonly_fields = ('created_at', 'updated_at', 'file_size_bytes', 'page_count', 'download_url')

    def download_url(self, obj):
        return obj.download_url
    download_url.short_description = 'Download URL'
