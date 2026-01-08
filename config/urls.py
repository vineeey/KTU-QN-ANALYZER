"""
URL configuration for PYQ Analyzer project.

GUEST ACCESS MODE:
- Primary user flow is GUEST (no login required)
- Admin and authenticated features are OPTIONAL
- Root URL serves guest upload page
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Admin panel (optional - for maintenance only)
    path('admin/', admin.site.urls),
    
    # GUEST WORKFLOW (PRIMARY - NO LOGIN REQUIRED)
    # Root URL serves guest upload page
    path('', include('apps.core.guest_views')),  # Includes upload, job status, downloads
    
    # OPTIONAL: Authenticated user features (if user chooses to register)
    # These are NOT required for core functionality
    path('users/', include('apps.users.urls')),
    path('subjects/', include('apps.subjects.urls')),
    path('papers/', include('apps.papers.urls')),
    path('questions/', include('apps.questions.urls')),
    path('rules/', include('apps.rules.urls')),
    path('analytics/', include('apps.analytics.urls')),
    path('reports/', include('apps.reports.urls')),
    
    # Legacy analysis URLs (kept for backward compatibility)
    path('analysis/', include('apps.analysis.urls')),
    
    # Core utilities
    path('core/', include('apps.core.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    if hasattr(settings, 'STATICFILES_DIRS') and settings.STATICFILES_DIRS:
        urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
