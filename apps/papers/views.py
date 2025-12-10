"""Views for paper upload and management."""
from django.views.generic import ListView, DetailView, CreateView, DeleteView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
import hashlib

from apps.subjects.models import Subject, Module
from .models import Paper
from .forms import PaperUploadForm, BatchPaperUploadForm


class GenericPaperUploadView(LoginRequiredMixin, FormView):
    """Upload papers without requiring a pre-existing subject."""
    
    form_class = BatchPaperUploadForm
    template_name = 'papers/paper_upload_generic.html'
    
    def get_or_create_subject(self, name, code, university):
        """Get or create a subject with modules for the user."""
        from apps.subjects.models import Module
        
        # Use provided name/code or defaults
        subject_name = name or 'My Question Papers'
        subject_code = code or 'DEFAULT'
        
        subject, created = Subject.objects.get_or_create(
            user=self.request.user,
            name=subject_name,
            defaults={
                'code': subject_code,
                'description': f'Question papers for {subject_name}',
                'university': university or 'KTU',
            }
        )
        
        # Update university if provided
        if university and subject.university != university:
            subject.university = university
            subject.save()
        
        # Create 5 modules if they don't exist (for KTU)
        if subject.modules.count() == 0:
            module_names = [
                'Module 1',
                'Module 2', 
                'Module 3',
                'Module 4',
                'Module 5'
            ]
            for i, mod_name in enumerate(module_names, 1):
                Module.objects.create(
                    subject=subject,
                    name=mod_name,
                    number=i,
                    description=f'Module {i} of {subject_name}',
                    weightage=20  # Equal weightage
                )
        
        self._subject = subject
        return subject
    
    def get_success_url(self):
        # Redirect to subject detail page to show progress
        if hasattr(self, '_subject') and self._subject:
            return reverse_lazy('subjects:detail', kwargs={'pk': self._subject.pk})
        return reverse_lazy('subjects:list')
    
    def post(self, request, *args, **kwargs):
        """Handle file upload."""
        files = request.FILES.getlist('files')
        university = request.POST.get('university', 'KTU')
        subject_name = request.POST.get('subject_name', '').strip()
        subject_code = request.POST.get('subject_code', '').strip()
        
        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return self.get(request, *args, **kwargs)
        
        # Validate files
        for f in files:
            if not f.name.lower().endswith('.pdf'):
                messages.error(request, f'"{f.name}" is not a PDF file. Only PDF files are supported.')
                return self.get(request, *args, **kwargs)
            if f.size > 50 * 1024 * 1024:
                messages.error(request, f'"{f.name}" exceeds 50MB limit.')
                return self.get(request, *args, **kwargs)
        
        # Get or create subject with modules
        subject = self.get_or_create_subject(subject_name, subject_code, university)
        
        # Process files
        uploaded_count = 0
        for uploaded_file in files:
            # Parse filename for metadata
            parsed = BatchPaperUploadForm.parse_filename(uploaded_file.name)
            
            # Calculate file hash for deduplication
            file_hash = hashlib.sha256()
            for chunk in uploaded_file.chunks():
                file_hash.update(chunk)
            file_hash_hex = file_hash.hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Check if file already exists (by hash)
            if Paper.objects.filter(subject=subject, file_hash=file_hash_hex).exists():
                messages.warning(
                    request,
                    f'"{uploaded_file.name}" already uploaded (skipped)'
                )
                continue
            
            # Create paper
            paper = Paper.objects.create(
                subject=subject,
                title=parsed['title'],
                year=parsed['year'],
                exam_type=parsed['exam_type'],
                file=uploaded_file,
                file_hash=file_hash_hex,
            )
            uploaded_count += 1
            
            # Queue background analysis task
            try:
                from apps.analysis.tasks import queue_paper_analysis
                queue_paper_analysis(paper)
            except Exception as e:
                messages.warning(
                    request,
                    f'"{uploaded_file.name}" uploaded but analysis could not be queued: {str(e)}'
                )
        
        if uploaded_count > 0:
            messages.success(
                request,
                f'{uploaded_count} paper(s) uploaded successfully! Analysis has been queued.'
            )
        
        return redirect(self.get_success_url())


class PaperListView(LoginRequiredMixin, ListView):
    """List papers for a subject."""
    
    model = Paper
    template_name = 'papers/paper_list.html'
    context_object_name = 'papers'
    
    def get_queryset(self):
        self.subject = get_object_or_404(
            Subject, pk=self.kwargs['subject_pk'], user=self.request.user
        )
        return Paper.objects.filter(subject=self.subject).order_by('-year', '-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['subject'] = self.subject
        return context


class PaperDetailView(LoginRequiredMixin, DetailView):
    """View paper details and extracted questions."""
    
    model = Paper
    template_name = 'papers/paper_detail.html'
    context_object_name = 'paper'
    
    def get_queryset(self):
        return Paper.objects.filter(subject__user=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['questions'] = self.object.questions.all().select_related('module')
        return context


class PaperUploadView(LoginRequiredMixin, FormView):
    """Upload multiple question papers at once."""
    
    form_class = BatchPaperUploadForm
    template_name = 'papers/paper_upload.html'
    
    def dispatch(self, request, *args, **kwargs):
        self.subject = get_object_or_404(
            Subject, pk=kwargs['subject_pk'], user=request.user
        )
        return super().dispatch(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['subject'] = self.subject
        return context
    
    def get_success_url(self):
        return reverse_lazy('papers:list', kwargs={'subject_pk': self.subject.pk})
    
    def post(self, request, *args, **kwargs):
        """Handle file upload - bypass form validation and handle files directly."""
        files = request.FILES.getlist('files')
        
        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return self.get(request, *args, **kwargs)
        
        # Validate files
        for f in files:
            if not f.name.lower().endswith('.pdf'):
                messages.error(request, f'"{f.name}" is not a PDF file. Only PDF files are supported.')
                return self.get(request, *args, **kwargs)
            if f.size > 50 * 1024 * 1024:
                messages.error(request, f'"{f.name}" exceeds 50MB limit.')
                return self.get(request, *args, **kwargs)
        
        # Process files
        uploaded_count = 0
        for uploaded_file in files:
            # Parse filename for metadata
            parsed = BatchPaperUploadForm.parse_filename(uploaded_file.name)
            
            # Calculate file hash for deduplication
            file_hash = hashlib.sha256()
            for chunk in uploaded_file.chunks():
                file_hash.update(chunk)
            file_hash_hex = file_hash.hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Check if file already exists (by hash)
            if Paper.objects.filter(subject=self.subject, file_hash=file_hash_hex).exists():
                messages.warning(
                    request,
                    f'"{uploaded_file.name}" already uploaded (skipped)'
                )
                continue
            
            # Create paper
            paper = Paper.objects.create(
                subject=self.subject,
                title=parsed['title'],
                year=parsed['year'],
                exam_type=parsed['exam_type'],
                file=uploaded_file,
                file_hash=file_hash_hex,
            )
            uploaded_count += 1
            
            # Queue background analysis task
            try:
                from apps.analysis.tasks import queue_paper_analysis
                queue_paper_analysis(paper)
            except Exception as e:
                messages.warning(
                    request,
                    f'"{uploaded_file.name}" uploaded but analysis could not be queued: {str(e)}'
                )
        
        if uploaded_count > 0:
            messages.success(
                request,
                f'{uploaded_count} paper(s) uploaded successfully! Analysis has been queued.'
            )
        
        return redirect(self.get_success_url())


class PaperDeleteView(LoginRequiredMixin, DeleteView):
    """Delete a paper."""
    
    model = Paper
    template_name = 'papers/paper_confirm_delete.html'
    
    def get_queryset(self):
        return Paper.objects.filter(subject__user=self.request.user)
    
    def get_success_url(self):
        return reverse_lazy('papers:list', kwargs={'subject_pk': self.object.subject.pk})
    
    def form_valid(self, form):
        messages.success(self.request, f'Paper "{self.object.title}" deleted.')
        return super().form_valid(form)
