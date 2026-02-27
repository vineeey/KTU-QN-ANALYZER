"""
Tests for HybridLLMService.
"""
import pytest
from unittest.mock import Mock, patch
from apps.analysis.services.hybrid_llm_service import HybridLLMService


class TestHybridLLMService:
    """Test suite for HybridLLMService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = HybridLLMService()
    
    def test_normalize_question_text(self):
        """Test question text normalization."""
        # Test question number removal
        text = "Q.1) Define stack data structure"
        normalized = self.service.normalize_question_text(text)
        assert normalized == "define stack data structure"
        
        # Test mark allocation removal
        text = "Explain binary search (14 marks)"
        normalized = self.service.normalize_question_text(text)
        assert normalized == "explain binary search"
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        self.service.stats['gemini_calls'] = 10
        self.service.stats['embedding_comparisons'] = 100
        
        stats = self.service.get_statistics()
        
        assert stats['total_llm_calls'] >= 0
        assert 'gemini' in stats
        assert 'similarity' in stats
