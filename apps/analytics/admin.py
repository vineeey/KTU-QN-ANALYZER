"""
Admin configuration for analytics app.
"""
from django.contrib import admin
from .models import TopicCluster, ClusterGroup, ClusterMembership, PriorityAssignment


@admin.register(TopicCluster)
class TopicClusterAdmin(admin.ModelAdmin):
    list_display = ('topic_name', 'subject', 'module', 'priority_tier', 'frequency_count', 'total_marks')
    list_filter = ('priority_tier', 'subject', 'module')
    search_fields = ('topic_name', 'subject__name', 'module__name')
    readonly_fields = ('created_at', 'updated_at')

    fieldsets = (
        ('Basic Information', {
            'fields': ('subject', 'module', 'topic_name', 'normalized_key')
        }),
        ('Statistics', {
            'fields': ('frequency_count', 'years_appeared', 'total_marks', 'priority_tier')
        }),
        ('Part Distribution', {
            'fields': ('part_a_count', 'part_b_count')
        }),
        ('Representative Text', {
            'fields': ('representative_text',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ClusterGroup)
class ClusterGroupAdmin(admin.ModelAdmin):
    list_display = ('cluster_label', 'subject', 'module', 'frequency', 'question_count', 'created_at')
    list_filter = ('subject', 'module')
    search_fields = ('subject__name', 'representative_text')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(ClusterMembership)
class ClusterMembershipAdmin(admin.ModelAdmin):
    list_display = ('cluster', 'question')
    list_filter = ('cluster__subject',)
    search_fields = ('question__text',)


@admin.register(PriorityAssignment)
class PriorityAssignmentAdmin(admin.ModelAdmin):
    list_display = ('cluster', 'tier', 'tier_label', 'frequency')
    list_filter = ('tier',)
    search_fields = ('cluster__subject__name',)
    readonly_fields = ('created_at', 'updated_at')
