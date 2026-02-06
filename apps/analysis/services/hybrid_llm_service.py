"""
Hybrid LLM Service for OCR Cleaning and Question Similarity Detection.
Uses Gemini 1.5 Flash as primary with Ollama as fallback.
"""
import logging
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from django.conf import settings
import numpy as np

logger = logging.getLogger(__name__)


class HybridLLMService:
    """
    Comprehensive hybrid LLM service with:
    - OCR text cleaning (Gemini primary, Ollama fallback)
    - Two-tier similarity detection (embeddings + LLM)
    - Statistics tracking
    """
    
    def __init__(self):
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        
        # Statistics tracking
        self.stats = {
            'gemini_calls': 0,
            'ollama_calls': 0,
            'gemini_failures': 0,
            'ollama_failures': 0,
            'embedding_comparisons': 0,
            'llm_comparisons': 0,
            'total_similarity_checks': 0,
        }
        
        # Gemini client (lazy loaded)
        self._gemini = None
        
        # Embedding model (lazy loaded)
        self._embedding_model = None
    
    def _get_gemini_client(self):
        """Lazy load Gemini client."""
        if self._gemini is None and self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self._gemini = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✓ Gemini 1.5 Flash initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        return self._gemini
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = settings.SIMILARITY_DETECTION.get('EMBEDDING_MODEL', 'multi-qa-MiniLM-L6-cos-v1')
                self._embedding_model = SentenceTransformer(model_name)
                logger.info(f"✓ Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Gemini API with retry logic.
        
        Args:
            prompt: The prompt to send
            max_retries: Maximum retry attempts
            
        Returns:
            Response text or None if failed
        """
        client = self._get_gemini_client()
        if not client:
            return None
        
        for attempt in range(max_retries):
            try:
                response = client.generate_content(prompt)
                self.stats['gemini_calls'] += 1
                return response.text
            except Exception as e:
                logger.warning(f"Gemini call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats['gemini_failures'] += 1
        
        return None
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API with retry logic (fallback).
        
        Args:
            prompt: The prompt to send
            max_retries: Maximum retry attempts
            
        Returns:
            Response text or None if failed
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed - cannot use Ollama")
            return None
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        f"{self.ollama_base_url}/api/generate",
                        json={
                            "model": self.ollama_model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    response.raise_for_status()
                    self.stats['ollama_calls'] += 1
                    return response.json()['response']
            except Exception as e:
                logger.warning(f"Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    self.stats['ollama_failures'] += 1
        
        return None
    
    def _call_llm(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Call LLM with automatic fallback from Gemini to Ollama.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Tuple of (response_text, llm_used)
            llm_used will be 'gemini', 'ollama', or 'none'
        """
        # Try Gemini first
        response = self._call_gemini(prompt)
        if response:
            return response, 'gemini'
        
        # Fallback to Ollama
        logger.info("Gemini unavailable, falling back to Ollama")
        response = self._call_ollama(prompt)
        if response:
            return response, 'ollama'
        
        return None, 'none'
    
    # ========== OCR CLEANING ==========
    
    def clean_ocr_text(
        self,
        raw_text: str,
        subject_name: Optional[str] = None,
        year: Optional[str] = None,
        use_advanced: bool = True
    ) -> Tuple[str, str]:
        """
        Clean OCR text using LLM.
        
        Args:
            raw_text: Raw OCR output
            subject_name: Optional subject context
            year: Optional year context
            use_advanced: Use advanced prompt with context
            
        Returns:
            Tuple of (cleaned_text, llm_used)
            llm_used will be 'gemini', 'ollama', or 'none'
        """
        if not settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', True):
            return raw_text, 'none'
        
        if use_advanced and subject_name:
            prompt = self._get_advanced_ocr_prompt(raw_text, subject_name, year)
        else:
            prompt = self._get_basic_ocr_prompt(raw_text)
        
        cleaned, llm_used = self._call_llm(prompt)
        
        if cleaned:
            return cleaned.strip(), llm_used
        else:
            logger.warning("LLM cleaning failed, returning original text")
            return raw_text, 'none'
    
    def clean_ocr_batch(
        self,
        pages: List[str],
        subject_name: Optional[str] = None,
        year: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        Clean multiple OCR pages in a single batch.
        
        Args:
            pages: List of raw OCR text per page
            subject_name: Optional subject context
            year: Optional year context
            
        Returns:
            Tuple of (cleaned_pages, llm_used)
        """
        if not settings.OCR_ENHANCEMENT.get('BATCH_PAGES', True):
            # Clean individually
            cleaned = []
            llm_used = 'none'
            for page in pages:
                cleaned_page, used = self.clean_ocr_text(page, subject_name, year)
                cleaned.append(cleaned_page)
                if used != 'none':
                    llm_used = used
            return cleaned, llm_used
        
        # Batch cleaning
        prompt = self._get_batch_ocr_prompt(pages, subject_name, year)
        cleaned, llm_used = self._call_llm(prompt)
        
        if cleaned:
            # Split back into pages
            pages_cleaned = self._split_batch_result(cleaned, len(pages))
            return pages_cleaned, llm_used
        else:
            logger.warning("Batch LLM cleaning failed, returning original pages")
            return pages, 'none'
    
    def _get_basic_ocr_prompt(self, text: str) -> str:
        """Generate basic OCR cleaning prompt."""
        return f"""You are an expert at cleaning OCR-extracted text from academic examination papers.

**Task:** Fix OCR errors while preserving the exact meaning and structure.

**Common OCR Errors to Fix:**
- Number confusion: l→1, O→0, S→5, I→1
- Question numbering: "l." → "1.", "Q.l" → "Q.1"
- Word breaks: "defi ne" → "define", "algo rithm" → "algorithm"
- Special characters: Replace garbled symbols with correct ones
- Spacing: Fix missing/extra spaces

**Rules:**
1. NEVER change the meaning or intent of questions
2. NEVER add or remove questions
3. Fix only OCR artifacts, not grammar or style
4. Preserve technical terms, formulas, and acronyms exactly
5. Maintain original numbering sequence

TEXT TO CLEAN:
{text}

Return ONLY the cleaned text, nothing else."""
    
    def _get_advanced_ocr_prompt(self, text: str, subject: str, year: Optional[str]) -> str:
        """Generate advanced OCR cleaning prompt with context."""
        year_context = f"\nYear: {year}" if year else ""
        return f"""You are an expert at cleaning OCR-extracted text from academic examination papers.

**Task:** Fix OCR errors while preserving the exact meaning and structure.

**Common OCR Errors to Fix:**
- Number confusion: l→1, O→0, S→5, I→1
- Question numbering: "l." → "1.", "Q.l" → "Q.1"
- Word breaks: "defi ne" → "define", "algo rithm" → "algorithm"
- Special characters: Replace garbled symbols with correct ones
- Spacing: Fix missing/extra spaces

**Rules:**
1. NEVER change the meaning or intent of questions
2. NEVER add or remove questions
3. Fix only OCR artifacts, not grammar or style
4. Preserve technical terms, formulas, and acronyms exactly
5. Maintain original numbering sequence

Subject: {subject}{year_context}

Raw OCR Text:
{text}

Return ONLY the cleaned text, nothing else."""
    
    def _get_batch_ocr_prompt(self, pages: List[str], subject: Optional[str], year: Optional[str]) -> str:
        """Generate batch OCR cleaning prompt for multiple pages."""
        page_text = "\n\n=== PAGE SEPARATOR ===\n\n".join(f"PAGE {i+1}:\n{page}" for i, page in enumerate(pages))
        
        subject_context = f"\nSubject: {subject}" if subject else ""
        year_context = f" ({year})" if year else ""
        
        return f"""Clean OCR errors in this multi-page KTU exam paper.{subject_context}{year_context}

Rules:
- Maintain page separators
- Preserve structure across pages
- Ensure consistent question numbering
- Handle questions spanning multiple pages
- Fix all OCR errors as specified

{page_text}

Return the cleaned pages with the same separator format."""
    
    def _split_batch_result(self, result: str, expected_pages: int) -> List[str]:
        """Split batch cleaning result back into individual pages."""
        # Split by page separator
        parts = re.split(r'===\s*PAGE SEPARATOR\s*===', result, flags=re.IGNORECASE)
        
        # Clean up and extract page content
        pages = []
        for part in parts:
            # Remove PAGE X: prefix if present
            cleaned = re.sub(r'PAGE\s+\d+\s*:\s*', '', part, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned:
                pages.append(cleaned)
        
        # Ensure we have the expected number of pages
        while len(pages) < expected_pages:
            pages.append("")
        
        return pages[:expected_pages]
    
    # ========== QUESTION SIMILARITY DETECTION ==========
    
    def normalize_question_text(self, text: str) -> str:
        """
        Normalize question text for comparison.
        
        Removes:
        - Question numbers and prefixes
        - Mark allocations
        - OR labels
        - Sub-question markers
        - Extra whitespace
        - Trailing punctuation
        
        Args:
            text: Raw question text
            
        Returns:
            Normalized text
        """
        # Remove question number prefix (Q1, Q.1, 1., etc.)
        text = re.sub(r'^(?:Q\.?\s*)?(\d+)[\.\)]\s*', '', text, flags=re.IGNORECASE)
        
        # Remove sub-question markers (a), (i), etc.
        text = re.sub(r'\([a-z0-9]+\)\s*', ' ', text, flags=re.IGNORECASE)
        
        # Remove mark allocations: (X marks), [X marks]
        text = re.sub(r'[\(\[]\s*\d+\s*marks?\s*[\)\]]', '', text, flags=re.IGNORECASE)
        
        # Remove OR label
        text = re.sub(r'\bOR\b', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove trailing punctuation
        text = text.strip().rstrip('.,;:?!')
        
        return text.strip()
    
    def compute_similarity_score(self, text1: str, text2: str) -> float:
        """
        Compute embedding-based similarity score between two questions.
        
        Args:
            text1: First question text
            text2: Second question text
            
        Returns:
            Similarity score (0-1)
        """
        model = self._get_embedding_model()
        
        # Normalize texts
        norm1 = self.normalize_question_text(text1)
        norm2 = self.normalize_question_text(text2)
        
        # Generate embeddings
        embeddings = model.encode([norm1, norm2], convert_to_numpy=True)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        self.stats['embedding_comparisons'] += 1
        self.stats['total_similarity_checks'] += 1
        
        return float(similarity)
    
    def are_questions_similar(
        self,
        question1: str,
        question2: str,
        marks1: Optional[int] = None,
        marks2: Optional[int] = None
    ) -> Tuple[bool, float, str, str]:
        """
        Two-tier similarity detection with hybrid approach.
        
        Tier 1: Fast embedding-based comparison
        Tier 2: LLM verification for edge cases
        
        Args:
            question1: First question text
            question2: Second question text
            marks1: Optional marks for first question
            marks2: Optional marks for second question
            
        Returns:
            Tuple of (is_similar, confidence, method, reason)
            - is_similar: Boolean
            - confidence: Score 0-1
            - method: 'embedding', 'llm', or 'hybrid'
            - reason: Explanation
        """
        if not settings.SIMILARITY_DETECTION.get('USE_HYBRID_APPROACH', True):
            # Fall back to simple embedding comparison
            score = self.compute_similarity_score(question1, question2)
            threshold = settings.SIMILARITY_DETECTION.get('THRESHOLD_HIGH', 0.85)
            is_similar = score >= threshold
            return is_similar, score, 'embedding', f'Embedding similarity: {score:.3f}'
        
        # Tier 1: Embedding-based fast check
        score = self.compute_similarity_score(question1, question2)
        
        threshold_high = settings.SIMILARITY_DETECTION.get('THRESHOLD_HIGH', 0.85)
        threshold_low = settings.SIMILARITY_DETECTION.get('THRESHOLD_LOW', 0.65)
        
        # Definitely similar
        if score >= threshold_high:
            return True, score, 'embedding', f'High similarity: {score:.3f}'
        
        # Definitely different
        if score < threshold_low:
            return False, score, 'embedding', f'Low similarity: {score:.3f}'
        
        # Edge case: Need LLM verification
        if settings.SIMILARITY_DETECTION.get('USE_LLM_FOR_EDGE_CASES', True):
            return self._llm_similarity_check(question1, question2, marks1, marks2, score)
        else:
            # Without LLM, use middle threshold
            threshold_mid = (threshold_high + threshold_low) / 2
            is_similar = score >= threshold_mid
            return is_similar, score, 'embedding', f'Mid-range similarity: {score:.3f}'
    
    def _llm_similarity_check(
        self,
        question1: str,
        question2: str,
        marks1: Optional[int],
        marks2: Optional[int],
        embedding_score: float
    ) -> Tuple[bool, float, str, str]:
        """
        Use LLM to verify similarity for edge cases.
        
        Returns:
            Tuple of (is_similar, confidence, method, reason)
        """
        prompt = self._get_similarity_prompt(question1, question2, marks1, marks2)
        response, llm_used = self._call_llm(prompt)
        
        self.stats['llm_comparisons'] += 1
        
        if not response:
            # LLM failed, fall back to embedding
            logger.warning("LLM similarity check failed, using embedding score")
            threshold_mid = (settings.SIMILARITY_DETECTION.get('THRESHOLD_HIGH', 0.85) + 
                           settings.SIMILARITY_DETECTION.get('THRESHOLD_LOW', 0.65)) / 2
            is_similar = embedding_score >= threshold_mid
            return is_similar, embedding_score, 'embedding', 'LLM unavailable, used embedding'
        
        # Parse LLM response
        try:
            is_similar, confidence, reason = self._parse_similarity_response(response)
            return is_similar, confidence, 'hybrid', reason
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            threshold_mid = (settings.SIMILARITY_DETECTION.get('THRESHOLD_HIGH', 0.85) + 
                           settings.SIMILARITY_DETECTION.get('THRESHOLD_LOW', 0.65)) / 2
            is_similar = embedding_score >= threshold_mid
            return is_similar, embedding_score, 'embedding', 'LLM parse error'
    
    def _get_similarity_prompt(
        self,
        q1: str,
        q2: str,
        marks1: Optional[int],
        marks2: Optional[int]
    ) -> str:
        """Generate question similarity detection prompt."""
        marks_info = ""
        if marks1 is not None and marks2 is not None:
            marks_info = f"\nQuestion 1 marks: {marks1}\nQuestion 2 marks: {marks2}\n"
        
        return f"""You are an expert at comparing academic examination questions.

**Task:** Determine if these two questions are asking about the SAME concept/topic.

Question 1: {q1}
Question 2: {q2}
{marks_info}

**Criteria for SIMILAR:**
- Testing the exact same concept/algorithm/theory
- Different wording but same learning objective
- One is a subset/superset of the other

**Criteria for DIFFERENT:**
- Different topics/concepts entirely
- Same domain but different specific topics
- Different depth levels (conceptual vs implementation)

Respond in this EXACT format:
VERDICT: [SIMILAR or DIFFERENT]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]

Be strict: When in doubt, mark as DIFFERENT."""
    
    def _get_advanced_similarity_prompt(
        self,
        q1: str,
        q2: str,
        marks1: Optional[int],
        marks2: Optional[int]
    ) -> str:
        """Generate advanced structured similarity analysis prompt."""
        marks_info = ""
        if marks1 is not None and marks2 is not None:
            marks_info = f'"marks1": {marks1}, "marks2": {marks2}, '
        
        return f"""Analyze if these two exam questions test the same knowledge/concept.

Perform step-by-step analysis:

1. Extract core topics from both questions
2. Compare question types (define/explain/derive/compare/write algorithm/draw/prove)
3. Compare scope and mark allocations
4. Make similarity decision

Question 1: {q1}
Question 2: {q2}

Return JSON format:
{{
  {marks_info}"topics_q1": ["topic1", "topic2"],
  "topics_q2": ["topic1", "topic2"],
  "type_q1": "explain|define|derive|compare|...",
  "type_q2": "explain|define|derive|compare|...",
  "similar": true|false,
  "confidence": 0.0-1.0,
  "reason": "explanation",
  "relationship": "identical|rewording|generalization|different_subtopic|completely_different"
}}"""
    
    def _parse_similarity_response(self, response: str) -> Tuple[bool, float, str]:
        """
        Parse LLM similarity response.
        
        Returns:
            Tuple of (is_similar, confidence, reason)
        """
        # Use the validator to parse the response
        from .llm_validator import LLMResponseValidator
        
        # Try new VERDICT format first
        try:
            is_similar, confidence, reason = LLMResponseValidator.parse_similarity_response(response)
            if reason != "Parse error":
                return is_similar, confidence, reason
        except Exception as e:
            logger.debug(f"Validator parsing failed, trying fallback: {e}")
        
        # Try to parse JSON format as fallback
        try:
            data = json.loads(response)
            is_similar = data.get('similar', False)
            confidence = float(data.get('confidence', 0.5))
            reason = data.get('reason', 'LLM analysis')
            return is_similar, confidence, reason
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            pass
        
        # Parse text format
        is_similar = False
        confidence = 0.5
        reason = "Unable to parse"
        
        # Look for SIMILAR: YES/NO
        if re.search(r'SIMILAR:\s*YES', response, re.IGNORECASE):
            is_similar = True
        elif re.search(r'SIMILAR:\s*NO', response, re.IGNORECASE):
            is_similar = False
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d\.]+)', response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Extract reason
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        
        return is_similar, confidence, reason
    
    def batch_similarity_check(
        self,
        new_question: str,
        existing_questions: List[Tuple[str, Optional[int]]],
        new_marks: Optional[int] = None
    ) -> List[Tuple[int, bool, float, str]]:
        """
        Check similarity of one question against multiple existing questions.
        Uses hybrid approach for each comparison.
        
        Args:
            new_question: The new question to check
            existing_questions: List of (question_text, marks) tuples
            new_marks: Optional marks for new question
            
        Returns:
            List of (index, is_similar, confidence, reason) for each existing question
        """
        results = []
        
        for idx, (existing_q, existing_marks) in enumerate(existing_questions):
            is_similar, confidence, method, reason = self.are_questions_similar(
                new_question,
                existing_q,
                new_marks,
                existing_marks
            )
            results.append((idx, is_similar, confidence, reason))
        
        return results
    
    # ========== STATISTICS ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_llm_calls = self.stats['gemini_calls'] + self.stats['ollama_calls']
        
        return {
            'gemini': {
                'calls': self.stats['gemini_calls'],
                'failures': self.stats['gemini_failures'],
                'success_rate': (
                    self.stats['gemini_calls'] / (self.stats['gemini_calls'] + self.stats['gemini_failures'])
                    if self.stats['gemini_calls'] + self.stats['gemini_failures'] > 0 else 0
                ),
            },
            'ollama': {
                'calls': self.stats['ollama_calls'],
                'failures': self.stats['ollama_failures'],
                'success_rate': (
                    self.stats['ollama_calls'] / (self.stats['ollama_calls'] + self.stats['ollama_failures'])
                    if self.stats['ollama_calls'] + self.stats['ollama_failures'] > 0 else 0
                ),
            },
            'similarity': {
                'embedding_only': self.stats['embedding_comparisons'] - self.stats['llm_comparisons'],
                'hybrid_llm': self.stats['llm_comparisons'],
                'total': self.stats['total_similarity_checks'],
                'llm_usage_rate': (
                    self.stats['llm_comparisons'] / self.stats['total_similarity_checks']
                    if self.stats['total_similarity_checks'] > 0 else 0
                ),
            },
            'total_llm_calls': total_llm_calls,
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
    
    def estimate_cost(self) -> Dict[str, Any]:
        """
        Estimate API usage cost.
        
        Gemini 1.5 Flash free tier: 1500 requests/day
        Returns cost estimation based on usage.
        """
        gemini_free_tier_limit = 1500
        gemini_remaining = max(0, gemini_free_tier_limit - self.stats['gemini_calls'])
        
        return {
            'gemini_calls_used': self.stats['gemini_calls'],
            'gemini_free_tier_remaining': gemini_remaining,
            'gemini_within_free_tier': self.stats['gemini_calls'] <= gemini_free_tier_limit,
            'ollama_calls': self.stats['ollama_calls'],
            'ollama_cost': 0,  # Local, no cost
            'note': 'Gemini 1.5 Flash offers 1500 free requests per day'
        }
