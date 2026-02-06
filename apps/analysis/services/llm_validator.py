"""Validate and sanitize LLM responses."""
import re
import logging

logger = logging.getLogger(__name__)


class LLMResponseValidator:
    """Validates LLM outputs for correctness and safety."""
    
    @staticmethod
    def validate_cleaned_text(original: str, cleaned: str) -> tuple[bool, str]:
        """Ensure cleaned text is valid and similar to original."""
        # Check if cleaned text is not empty
        if not cleaned or len(cleaned.strip()) < 10:
            return False, "Cleaned text is too short or empty"
        
        # Check if question count is preserved (rough heuristic)
        original_q_count = len(re.findall(r'\b[Qq]\w*\s*\d+', original))
        cleaned_q_count = len(re.findall(r'\b[Qq]\w*\s*\d+', cleaned))
        
        if abs(original_q_count - cleaned_q_count) > 2:
            return False, f"Question count mismatch: {original_q_count} vs {cleaned_q_count}"
        
        # Check length similarity (shouldn't deviate too much)
        len_ratio = len(cleaned) / len(original) if len(original) > 0 else 0
        if len_ratio < 0.5 or len_ratio > 2.0:
            return False, f"Length deviation too high: {len_ratio:.2f}x"
        
        return True, "Valid"
    
    @staticmethod
    def parse_similarity_response(response: str) -> tuple[bool, float, str]:
        """Parse LLM similarity response with strict validation."""
        try:
            # Extract verdict
            verdict_match = re.search(r'VERDICT:\s*(SIMILAR|DIFFERENT)', response, re.IGNORECASE)
            if not verdict_match:
                logger.warning("No valid verdict found in LLM response")
                return False, 0.0, "Parse error"
            
            verdict = verdict_match.group(1).upper()
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
            confidence = float(conf_match.group(1)) / 100 if conf_match else 0.5
            
            # Extract reasoning
            reason_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"
            
            is_similar = verdict == "SIMILAR"
            
            return is_similar, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Failed to parse similarity response: {e}")
            return False, 0.0, f"Parse error: {e}"
    
    @staticmethod
    def validate_module_classification(question: str, module_num: int, num_modules: int) -> bool:
        """Validate module classification is within valid range."""
        if not isinstance(module_num, int):
            return False
        if module_num < 1 or module_num > num_modules:
            return False
        return True
