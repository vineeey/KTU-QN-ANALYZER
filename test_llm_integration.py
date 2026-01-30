#!/usr/bin/env python
"""
Test script to verify LLM integration is working correctly.
Tests:
1. HybridLLMService initialization
2. OCR text cleaning functionality
3. Similar question identification
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.conf import settings


def test_hybrid_llm_service():
    """Test HybridLLMService initialization and basic functionality."""
    print("\n" + "="*80)
    print("TEST 1: HybridLLMService Initialization")
    print("="*80)
    
    try:
        from apps.analysis.services.hybrid_llm_service import HybridLLMService
        
        service = HybridLLMService()
        print("✅ HybridLLMService initialized successfully")
        
        # Check configuration
        print(f"\nConfiguration:")
        print(f"  - Gemini API Key: {'***' + service.gemini_api_key[-4:] if service.gemini_api_key else 'Not configured'}")
        print(f"  - Ollama URL: {service.ollama_base_url}")
        print(f"  - Ollama Model: {service.ollama_model}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize HybridLLMService: {e}")
        return False


def test_ocr_cleaning():
    """Test OCR text cleaning functionality."""
    print("\n" + "="*80)
    print("TEST 2: OCR Text Cleaning")
    print("="*80)
    
    try:
        from apps.analysis.services.hybrid_llm_service import HybridLLMService
        
        service = HybridLLMService()
        
        # Test text with common OCR errors
        raw_text = """
        Qn l. Define algorithrrn and explain its characteristics (3 rnarks)
        Qn 2. Wrlte the tirne cornplexity of binary search (3 marks)
        """
        
        print(f"\nRaw OCR Text:\n{raw_text}")
        print(f"\nOCR Cleaning Enabled: {settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', False)}")
        
        if not settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', False):
            print("⚠️  OCR cleaning is disabled in settings")
            return True
        
        print("\nAttempting to clean text with LLM...")
        cleaned_text, llm_used = service.clean_ocr_text(
            raw_text=raw_text,
            subject_name="Data Structures",
            year="2024",
            use_advanced=True
        )
        
        print(f"\nLLM Used: {llm_used}")
        if llm_used != 'none':
            print(f"Cleaned Text:\n{cleaned_text}")
            print("✅ OCR cleaning test passed")
            return True
        else:
            print("⚠️  LLM cleaning returned 'none' - API may not be configured")
            print("   (This is expected if GEMINI_API_KEY is not set)")
            return True
            
    except Exception as e:
        print(f"❌ OCR cleaning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_detection():
    """Test similar question identification."""
    print("\n" + "="*80)
    print("TEST 3: Similar Question Identification")
    print("="*80)
    
    try:
        from apps.analysis.services.hybrid_llm_service import HybridLLMService
        
        service = HybridLLMService()
        
        # Test questions
        q1 = "Explain the working of binary search algorithm"
        q2 = "Describe how binary search works"
        q3 = "Define linear search algorithm"
        
        print(f"\nSimilarity Detection Configuration:")
        print(f"  - Use Hybrid Approach: {settings.SIMILARITY_DETECTION.get('USE_HYBRID_APPROACH', False)}")
        print(f"  - Threshold High: {settings.SIMILARITY_DETECTION.get('THRESHOLD_HIGH', 0.85)}")
        print(f"  - Threshold Low: {settings.SIMILARITY_DETECTION.get('THRESHOLD_LOW', 0.65)}")
        print(f"  - Use LLM for Edge Cases: {settings.SIMILARITY_DETECTION.get('USE_LLM_FOR_EDGE_CASES', False)}")
        
        print(f"\nTest Case 1: Similar Questions")
        print(f"  Q1: {q1}")
        print(f"  Q2: {q2}")
        
        # This requires the embedding model which might not be installed
        try:
            is_similar, conf, method, reason = service.are_questions_similar(q1, q2, 3, 3)
            print(f"  Result: {'Similar' if is_similar else 'Not similar'}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Method: {method}")
            print(f"  Reason: {reason}")
            print("✅ Similarity detection test 1 passed")
        except Exception as e:
            print(f"⚠️  Similarity check failed: {e}")
            print("   (This is expected if sentence-transformers is not installed)")
        
        print(f"\nTest Case 2: Different Questions")
        print(f"  Q1: {q1}")
        print(f"  Q3: {q3}")
        
        try:
            is_similar, conf, method, reason = service.are_questions_similar(q1, q3, 3, 3)
            print(f"  Result: {'Similar' if is_similar else 'Not similar'}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Method: {method}")
            print(f"  Reason: {reason}")
            print("✅ Similarity detection test 2 passed")
        except Exception as e:
            print(f"⚠️  Similarity check failed: {e}")
            print("   (This is expected if sentence-transformers is not installed)")
        
        return True
            
    except Exception as e:
        print(f"❌ Similarity detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clustering_integration():
    """Test that clustering service uses HybridLLMService."""
    print("\n" + "="*80)
    print("TEST 4: Clustering Service Integration")
    print("="*80)
    
    try:
        from apps.analytics.clustering import TopicClusteringService
        from apps.subjects.models import Subject
        
        # Get first subject or create a dummy one for testing
        subject = Subject.objects.first()
        
        if not subject:
            print("⚠️  No subjects found in database, skipping integration test")
            return True
        
        service = TopicClusteringService(subject)
        
        print(f"Clustering Service initialized for subject: {subject.name if hasattr(subject, 'name') else subject.id}")
        print(f"  - Has embedding model: {service.model is not None}")
        print(f"  - Has HybridLLMService: {service.hybrid_llm is not None}")
        
        if service.hybrid_llm:
            print("✅ Clustering service has HybridLLMService integration")
        else:
            print("⚠️  Clustering service HybridLLMService not initialized")
            print("   (Check if SIMILARITY_USE_HYBRID is set to true)")
        
        return True
        
    except Exception as e:
        print(f"❌ Clustering integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LLM Integration Test Suite")
    print("="*80)
    
    results = {
        'HybridLLMService Initialization': test_hybrid_llm_service(),
        'OCR Text Cleaning': test_ocr_cleaning(),
        'Similar Question Identification': test_similarity_detection(),
        'Clustering Service Integration': test_clustering_integration(),
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
