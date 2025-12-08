"""Views for analytics dashboard."""
from django.views.generic import TemplateView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib import messages
from pathlib import Path

from apps.subjects.models import Subject, Module
from apps.analytics.models import TopicCluster
from .calculator import StatsCalculator


class AnalyticsDashboardView(LoginRequiredMixin, TemplateView):
    """Analytics dashboard for a subject showing all modules."""
    
    template_name = 'analytics/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        subject = get_object_or_404(
            Subject, pk=self.kwargs['subject_pk'], user=self.request.user
        )
        
        calculator = StatsCalculator(subject)
        stats = calculator.get_complete_stats()
        
        # Get top topics per module for the main graph
        top_topics = calculator.get_top_topics_per_module(top_n=3)
        
        # Get all modules with their topic counts
        modules = subject.modules.all()
        module_data = []
        for module in modules:
            topic_count = TopicCluster.objects.filter(
                subject=subject,
                module=module
            ).count()
            tier_1_count = TopicCluster.objects.filter(
                subject=subject,
                module=module,
                priority_tier=TopicCluster.PriorityTier.TIER_1
            ).count()
            
            module_data.append({
                'module': module,
                'topic_count': topic_count,
                'critical_topics': tier_1_count,
                'top_topics': top_topics.get(module.number, [])
            })
        
        context['subject'] = subject
        context['stats'] = stats
        context['modules'] = module_data
        context['has_analysis'] = TopicCluster.objects.filter(subject=subject).exists()
        
        return context


class ModuleAnalyticsView(LoginRequiredMixin, TemplateView):
    """Detailed analytics for a specific module."""
    
    template_name = 'analytics/module_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        subject = get_object_or_404(
            Subject, pk=self.kwargs['subject_pk'], user=self.request.user
        )
        module_number = self.kwargs['module_number']
        module = get_object_or_404(Module, subject=subject, number=module_number)
        
        calculator = StatsCalculator(subject)
        module_stats = calculator.get_module_topic_stats(module_number)
        
        # Get all topic clusters for this module
        topics = TopicCluster.objects.filter(
            subject=subject,
            module=module
        ).order_by('-frequency_count')
        
        # Count topics by tier
        tier_counts = {
            'tier_1': topics.filter(priority_tier=TopicCluster.PriorityTier.TIER_1).count(),
            'tier_2': topics.filter(priority_tier=TopicCluster.PriorityTier.TIER_2).count(),
            'tier_3': topics.filter(priority_tier=TopicCluster.PriorityTier.TIER_3).count(),
            'tier_4': topics.filter(priority_tier=TopicCluster.PriorityTier.TIER_4).count(),
        }
        
        context['subject'] = subject
        context['module'] = module
        context['stats'] = module_stats
        context['topics'] = topics
        context['tier_counts'] = tier_counts
        
        return context


class TriggerTopicAnalysisView(LoginRequiredMixin, View):
    """Trigger topic clustering analysis for a subject."""
    
    def post(self, request, subject_pk):
        subject = get_object_or_404(
            Subject, pk=subject_pk, user=request.user
        )
        
        # Queue the analysis task
        try:
            from apps.analysis.tasks import queue_topic_analysis
            queue_topic_analysis(subject)
            messages.success(
                request,
                'Topic analysis has been queued. This may take a few minutes.'
            )
        except Exception as e:
            messages.error(
                request,
                f'Failed to queue analysis: {str(e)}'
            )
        
        return redirect('analytics:dashboard', subject_pk=subject.pk)


class AnalyticsAPIView(LoginRequiredMixin, View):
    """API endpoint for analytics data (for Chart.js)."""
    
    def get(self, request, subject_pk):
        subject = get_object_or_404(
            Subject, pk=subject_pk, user=request.user
        )
        
        calculator = StatsCalculator(subject)
        stats = calculator.get_complete_stats()
        
        # Format data for charts
        chart_data = {
            'overview': stats['overview'],
            'modules': [],
            'top_topics': []
        }
        
        # Module distribution for pie chart
        for mod_data in stats['module_distribution']:
            chart_data['modules'].append({
                'label': f"Module {mod_data['module_number']}: {mod_data['module']}" if mod_data['module_number'] else 'Unclassified',
                'count': mod_data['count']
            })
        
        # Top topics per module for bar chart
        top_topics = stats.get('top_topics_per_module', {})
        for module_num, topics in top_topics.items():
            for topic in topics:
                chart_data['top_topics'].append({
                    'module': f"M{module_num}",
                    'topic': topic['topic'][:30] + '...' if len(topic['topic']) > 30 else topic['topic'],
                    'frequency': topic['frequency'],
                    'priority': topic['priority']
                })
        
        return JsonResponse(chart_data)
