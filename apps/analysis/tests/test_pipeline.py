"""Integration tests for analysis pipeline."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.subjects.models import Subject, Module
from apps.papers.models import Paper
from apps.analysis.pipeline import AnalysisPipeline


@pytest.mark.django_db
class TestAnalysisPipeline:
    """Test complete analysis workflow."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized without errors."""
        pipeline = AnalysisPipeline()
        assert pipeline is not None
        assert pipeline.pymupdf_extractor is not None
        assert pipeline.fallback_extractor is not None
    
    def test_subject_and_module_creation(self):
        """Test creating subject with modules for testing."""
        # Create test subject
        subject = Subject.objects.create(
            name='Data Structures',
            code='CS201',
            university_type='KTU'
        )
        
        # Create modules
        modules = []
        for i in range(1, 6):
            module = Module.objects.create(
                subject=subject,
                name=f'Module {i}',
                number=i
            )
            modules.append(module)
        
        # Verify creation
        assert subject.modules.count() == 5
        assert all(m.subject == subject for m in modules)
    
    def test_paper_creation(self):
        """Test creating a paper object."""
        subject = Subject.objects.create(
            name='Test Subject',
            code='TEST101',
            university_type='KTU'
        )
        
        # Create a minimal PDF-like file
        pdf_content = b'%PDF-1.4\nTest PDF content'
        pdf_file = SimpleUploadedFile(
            'test.pdf',
            pdf_content,
            content_type='application/pdf'
        )
        
        paper = Paper.objects.create(
            subject=subject,
            title='Test Paper',
            file=pdf_file,
            status='pending'
        )
        
        assert paper.id is not None
        assert paper.subject == subject
        assert paper.status == 'pending'
    
    def test_question_validation(self):
        """Test question model validation."""
        from apps.questions.models import Question
        from django.core.exceptions import ValidationError
        
        subject = Subject.objects.create(
            name='Test Subject',
            code='TEST101'
        )
        
        pdf_content = b'%PDF-1.4\nTest'
        pdf_file = SimpleUploadedFile('test.pdf', pdf_content)
        
        paper = Paper.objects.create(
            subject=subject,
            title='Test Paper',
            file=pdf_file
        )
        
        # Test valid question
        question = Question(
            paper=paper,
            text='What is a data structure?',
            marks=5,
            question_number='1'
        )
        question.full_clean()  # Should not raise
        question.save()
        
        # Test invalid marks (should raise)
        invalid_question = Question(
            paper=paper,
            text='Test question',
            marks=-5,
            question_number='2'
        )
        
        with pytest.raises(ValidationError):
            invalid_question.full_clean()
        
        # Test short text (should raise)
        short_question = Question(
            paper=paper,
            text='ABC',
            marks=5,
            question_number='3'
        )
        
        with pytest.raises(ValidationError):
            short_question.full_clean()


@pytest.mark.django_db
class TestLLMValidator:
    """Test LLM response validator."""
    
    def test_validate_cleaned_text(self):
        """Test OCR text validation."""
        from apps.analysis.services.llm_validator import LLMResponseValidator
        
        # Valid case
        original = "Q1. What is a data structure?"
        cleaned = "Q1. What is a data structure?"
        is_valid, msg = LLMResponseValidator.validate_cleaned_text(original, cleaned)
        assert is_valid is True
        
        # Too short
        is_valid, msg = LLMResponseValidator.validate_cleaned_text(original, "Q1")
        assert is_valid is False
        assert "too short" in msg
        
        # Question count mismatch
        is_valid, msg = LLMResponseValidator.validate_cleaned_text(
            "Q1. First\nQ2. Second\nQ3. Third",
            "Q1. Only one"
        )
        assert is_valid is False
        assert "mismatch" in msg
    
    def test_parse_similarity_response(self):
        """Test similarity response parsing."""
        from apps.analysis.services.llm_validator import LLMResponseValidator
        
        # Valid VERDICT format
        response = """VERDICT: SIMILAR
CONFIDENCE: 85
REASONING: Both questions test the same concept"""
        
        is_similar, confidence, reasoning = LLMResponseValidator.parse_similarity_response(response)
        assert is_similar is True
        assert confidence == 0.85
        assert "same concept" in reasoning
        
        # DIFFERENT verdict
        response2 = """VERDICT: DIFFERENT
CONFIDENCE: 70
REASONING: Different topics entirely"""
        
        is_similar2, conf2, reason2 = LLMResponseValidator.parse_similarity_response(response2)
        assert is_similar2 is False
        assert conf2 == 0.70
    
    def test_validate_module_classification(self):
        """Test module classification validation."""
        from apps.analysis.services.llm_validator import LLMResponseValidator
        
        # Valid
        assert LLMResponseValidator.validate_module_classification("Q", 1, 5) is True
        assert LLMResponseValidator.validate_module_classification("Q", 5, 5) is True
        
        # Invalid - out of range
        assert LLMResponseValidator.validate_module_classification("Q", 0, 5) is False
        assert LLMResponseValidator.validate_module_classification("Q", 6, 5) is False
        
        # Invalid - not int
        assert LLMResponseValidator.validate_module_classification("Q", "1", 5) is False


@pytest.mark.django_db
class TestPDFValidation:
    """Test PDF upload validation."""
    
    def test_validate_pdf_valid(self):
        """Test PDF validation with valid file."""
        from apps.papers.views import validate_pdf
        
        # Valid PDF
        pdf_content = b'%PDF-1.4\n' + b'x' * 2048
        pdf_file = SimpleUploadedFile(
            'test.pdf',
            pdf_content,
            content_type='application/pdf'
        )
        
        is_valid, msg = validate_pdf(pdf_file)
        assert is_valid is True
        assert msg == "Valid"
    
    def test_validate_pdf_wrong_extension(self):
        """Test rejection of non-PDF extension."""
        from apps.papers.views import validate_pdf
        
        file = SimpleUploadedFile(
            'test.txt',
            b'Not a PDF',
            content_type='text/plain'
        )
        
        is_valid, msg = validate_pdf(file)
        assert is_valid is False
        assert ".pdf" in msg
    
    def test_validate_pdf_too_large(self):
        """Test rejection of oversized files."""
        from apps.papers.views import validate_pdf
        
        # Create a file larger than 50MB
        large_content = b'%PDF-1.4\n' + b'x' * (51 * 1024 * 1024)
        large_file = SimpleUploadedFile(
            'large.pdf',
            large_content,
            content_type='application/pdf'
        )
        
        is_valid, msg = validate_pdf(large_file)
        assert is_valid is False
        assert "50MB" in msg
    
    def test_validate_pdf_too_small(self):
        """Test rejection of too-small files."""
        from apps.papers.views import validate_pdf
        
        tiny_file = SimpleUploadedFile(
            'tiny.pdf',
            b'%PDF',
            content_type='application/pdf'
        )
        
        is_valid, msg = validate_pdf(tiny_file)
        assert is_valid is False
        assert "too small" in msg
