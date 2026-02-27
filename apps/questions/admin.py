from django.contrib import admin
from .models import Question, QuestionEmbeddingCache


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_number', 'paper', 'module', 'difficulty', 'bloom_level', 'is_duplicate')
    list_filter = ('difficulty', 'bloom_level', 'is_duplicate', 'module')
    search_fields = ('text', 'paper__title')


@admin.register(QuestionEmbeddingCache)
class QuestionEmbeddingCacheAdmin(admin.ModelAdmin):
    list_display = ('question', 'model_name', 'vector_dim', 'updated_at')
    list_filter = ('model_name',)
    readonly_fields = ('created_at', 'updated_at')

    def vector_dim(self, obj):
        return len(obj.vector) if obj.vector else 0
    vector_dim.short_description = 'Vector Dim'
