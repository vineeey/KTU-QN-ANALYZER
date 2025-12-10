"""URL patterns for papers app."""
from django.urls import path
from . import views

app_name = 'papers'

urlpatterns = [
    # Public upload - no authentication required
    path('upload/', views.GenericPaperUploadView.as_view(), name='upload_generic'),
    
    # Subject-specific views
    path('subject/<uuid:subject_pk>/', views.PaperListView.as_view(), name='list'),
    path('subject/<uuid:subject_pk>/upload/', views.PaperUploadView.as_view(), name='upload'),
    path('<uuid:pk>/', views.PaperDetailView.as_view(), name='detail'),
    path('<uuid:pk>/delete/', views.PaperDeleteView.as_view(), name='delete'),
]
