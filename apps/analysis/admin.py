from django.contrib import admin
from .models import AnalysisJob
from .job_models import TempPaper, TempQuestion, TempTopicCluster


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'subject_name', 'status', 'progress', 'questions_extracted', 'created_at')
    list_filter = ('status',)
    search_fields = ('subject_name', 'id')
    readonly_fields = ('id', 'created_at', 'completed_at', 'expires_at')


@admin.register(TempPaper)
class TempPaperAdmin(admin.ModelAdmin):
    list_display = ('filename', 'job', 'year', 'pdf_type', 'uploaded_at')
    list_filter = ('pdf_type',)
    search_fields = ('filename', 'job__subject_name')


@admin.register(TempQuestion)
class TempQuestionAdmin(admin.ModelAdmin):
    list_display = ('question_number', 'paper', 'part', 'marks', 'module_number')
    list_filter = ('part', 'module_number')
    search_fields = ('raw_text',)


@admin.register(TempTopicCluster)
class TempTopicClusterAdmin(admin.ModelAdmin):
    list_display = ('topic_label', 'module_number', 'priority_tier', 'confidence_score', 'frequency')
    list_filter = ('module_number', 'priority_tier')
    search_fields = ('topic_label',)
