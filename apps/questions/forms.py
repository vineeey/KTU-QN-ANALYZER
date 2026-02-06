"""Forms for question editing."""
from django import forms
from .models import Question
from apps.subjects.models import Module


class QuestionEditForm(forms.ModelForm):
    """Form for manually editing question classification."""
    
    class Meta:
        model = Question
        fields = ['module', 'difficulty', 'bloom_level', 'marks']
        widgets = {
            'module': forms.Select(attrs={'class': 'form-select'}),
            'difficulty': forms.Select(attrs={'class': 'form-select'}),
            'bloom_level': forms.Select(attrs={'class': 'form-select'}),
            'marks': forms.NumberInput(attrs={'class': 'form-input', 'min': 0}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filter modules to only show those belonging to the question's subject
        if self.instance and self.instance.paper:
            self.fields['module'].queryset = Module.objects.filter(
                subject=self.instance.paper.subject
            )
