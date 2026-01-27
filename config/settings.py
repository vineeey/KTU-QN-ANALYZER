"""
Django settings for PYQ Analyzer project.
Zero-cost edition with SQLite3 and Qwen 2.5 7B Instruct API.
"""
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    # python-dotenv not installed, will use system env vars
    pass

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'django-insecure-change-this-in-production-pyq-analyzer-secret-key'
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'django_extensions',
    'django_q',
    
    # Local apps
    'apps.core',
    'apps.users',
    'apps.subjects',
    'apps.papers',
    'apps.questions',
    'apps.rules',
    'apps.analysis',
    'apps.analytics',
    'apps.reports',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'apps.core.context_processors.global_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database - SQLite3 with WAL mode for better concurrency
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db' / 'pyq_analyzer.sqlite3',
        'OPTIONS': {
            'timeout': 30,
        },
    }
}

# Custom User Model (optional now for public access)
AUTH_USER_MODEL = 'users.User'

# Authentication backends (email-based login)
AUTHENTICATION_BACKENDS = [
    'apps.users.backends.EmailBackend',
    'django.contrib.auth.backends.ModelBackend',
]

# Login/Logout URLs - optional for public mode
LOGIN_URL = 'users:login'
LOGIN_REDIRECT_URL = 'core:dashboard'
LOGOUT_REDIRECT_URL = 'core:home'

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (User uploads)
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django-Q2 Configuration (SQLite-based task queue)
Q_CLUSTER = {
    'name': 'pyq_analyzer',
    'workers': 2,
    'recycle': 500,
    'timeout': 1800,  # 30 minutes for LLM processing
    'retry': 2000,
    'compress': True,
    'save_limit': 250,
    'queue_limit': 500,
    'cpu_affinity': 1,
    'label': 'Django Q2',
    'orm': 'default',
}

# Qwen API Configuration
QWEN_API_KEY = os.environ.get('QWEN_API_KEY', '')
QWEN_API_URL = os.environ.get('QWEN_API_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
QWEN_MODEL = os.environ.get('QWEN_MODEL', 'qwen2.5-7b-instruct')

# Gemini API Configuration (Primary LLM for OCR and similarity)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Ollama Configuration (Fallback for Gemini)
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:7b-instruct')
# Supported models: qwen2.5:7b-instruct, llama3.2:3b, llama3.1:3b, phi3, tinyllama
OLLAMA_TIMEOUT = int(os.environ.get('OLLAMA_TIMEOUT', '120'))

# Embedding Model Configuration
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Universal Exam Analyzer Settings
UNIVERSAL_EXAM_ANALYZER = {
    'PUBLIC_ACCESS': True,  # No authentication required
    'KTU_RULE_BASED': True,  # Use strict rules for KTU
    'AI_CLASSIFICATION': True,  # Use AI for other universities
    'ENABLE_OCR': True,  # Enable Tesseract OCR for scanned PDFs
    'EXTRACT_IMAGES': True,  # Extract images with PyMuPDF
    'USE_LOCAL_LLM': True,  # Use Ollama for LLM tasks
    'MIN_CONFIDENCE': 0.6,  # Minimum confidence for AI classification
}

# OCR Enhancement Settings
OCR_ENHANCEMENT = {
    'USE_LLM_CLEANING': os.environ.get('OCR_USE_LLM_CLEANING', 'true').lower() in ('true', '1', 'yes'),
    'USE_ADVANCED_PREPROCESSING': True,
    'BATCH_PAGES': os.environ.get('OCR_BATCH_PAGES', 'true').lower() in ('true', '1', 'yes'),
    'MAX_PAGES_PER_BATCH': 5,
}

# Similarity Detection Settings
SIMILARITY_DETECTION = {
    'USE_HYBRID_APPROACH': os.environ.get('SIMILARITY_USE_HYBRID', 'true').lower() in ('true', '1', 'yes'),
    'EMBEDDING_MODEL': os.environ.get('SIMILARITY_EMBEDDING_MODEL', 'multi-qa-MiniLM-L6-cos-v1'),
    'THRESHOLD_HIGH': float(os.environ.get('SIMILARITY_THRESHOLD_HIGH', '0.85')),
    'THRESHOLD_LOW': float(os.environ.get('SIMILARITY_THRESHOLD_LOW', '0.65')),
    'USE_LLM_FOR_EDGE_CASES': True,
}

# File Upload Settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB for PDFs
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'pyq_analyzer.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
        'apps': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Create necessary directories
for directory in ['db', 'logs', 'media/papers', 'media/syllabi', 'media/reports']:
    (BASE_DIR / directory).mkdir(parents=True, exist_ok=True)
