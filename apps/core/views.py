"""
Core views - Home and Dashboard.
"""
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count

from apps.papers.models import Paper
from apps.questions.models import Question


class HomeView(TemplateView):
    """Landing page view - Public access."""
    template_name = 'pages/home_new.html'  # New 3D animated homepage


class DashboardView(TemplateView):
    """Main dashboard view - Public access enabled."""
    template_name = 'pages/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get user's statistics (if authenticated)
        if user.is_authenticated:
            context['total_subjects'] = user.subjects.count() if hasattr(user, 'subjects') else 0
            context['total_papers'] = Paper.objects.filter(subject__user=user).count()
            context['total_questions'] = Question.objects.filter(paper__subject__user=user).count()
            
            # Get recent subjects
            context['recent_subjects'] = user.subjects.all()[:5] if hasattr(user, 'subjects') else []
        else:
            # Public view - show general stats
            context['total_subjects'] = 0
            context['total_papers'] = 0
            context['total_questions'] = 0
            context['recent_subjects'] = []
        
        return context
