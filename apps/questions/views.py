"""Views for question management."""
from django.views.generic import ListView, DetailView, UpdateView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.shortcuts import redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
import csv

from .models import Question
from .forms import QuestionEditForm


class QuestionListView(LoginRequiredMixin, ListView):
    """List all questions for a paper."""
    
    model = Question
    template_name = 'questions/question_list.html'
    context_object_name = 'questions'
    
    def get_queryset(self):
        qs = Question.objects.filter(
            paper__subject__user=self.request.user
        ).select_related('paper', 'module')
        
        # Filter by paper if specified
        paper_id = self.request.GET.get('paper')
        if paper_id:
            qs = qs.filter(paper_id=paper_id)
        
        # Filter by subject if specified
        subject_id = self.request.GET.get('subject')
        if subject_id:
            qs = qs.filter(paper__subject_id=subject_id)
        
        return qs


class QuestionDetailView(LoginRequiredMixin, DetailView):
    """View question details."""
    
    model = Question
    template_name = 'questions/question_detail.html'
    context_object_name = 'question'
    
    def get_queryset(self):
        return Question.objects.filter(
            paper__subject__user=self.request.user
        ).select_related('paper', 'module', 'duplicate_of')


class QuestionUpdateView(LoginRequiredMixin, UpdateView):
    """Update question classification."""
    
    model = Question
    form_class = QuestionEditForm
    template_name = 'questions/question_edit.html'
    
    def get_queryset(self):
        return Question.objects.filter(paper__subject__user=self.request.user)
    
    def get_success_url(self):
        return reverse_lazy('questions:detail', kwargs={'pk': self.object.pk})
    
    def form_valid(self, form):
        # Set manual override flags
        if 'module' in form.changed_data:
            form.instance.module_manually_set = True
        if 'difficulty' in form.changed_data:
            form.instance.difficulty_manually_set = True
        messages.success(self.request, 'Question updated successfully!')
        return super().form_valid(form)


class QuestionVerifyView(LoginRequiredMixin, View):
    """Verify a question (mark as manually reviewed)."""
    
    def post(self, request, pk):
        question = get_object_or_404(
            Question,
            pk=pk,
            paper__subject__user=request.user
        )
        question.module_manually_set = True
        question.save()
        messages.success(request, 'Question verified successfully!')
        return redirect('questions:detail', pk=pk)


class QuestionExportView(LoginRequiredMixin, View):
    """Export questions to CSV."""
    
    def get(self, request):
        # Get filter parameters
        paper_id = request.GET.get('paper')
        subject_id = request.GET.get('subject')
        
        qs = Question.objects.filter(
            paper__subject__user=request.user
        ).select_related('paper', 'module')
        
        if paper_id:
            qs = qs.filter(paper_id=paper_id)
        if subject_id:
            qs = qs.filter(paper__subject_id=subject_id)
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="questions_export.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Question Number', 'Text', 'Module', 'Part', 'Marks', 'Paper', 'Subject'])
        
        for question in qs:
            writer.writerow([
                question.question_number,
                question.text[:200],
                question.module.name if question.module else '',
                question.part,
                question.marks,
                question.paper.title,
                question.paper.subject.name
            ])
        
        return response
