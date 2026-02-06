# LLM Configuration Guide

## API Keys Setup

### Gemini (Primary - FREE)
1. Get API key: https://makersuite.google.com/app/apikey
2. Add to `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Free tier: 1,500 requests/day

### Qwen (Secondary)
1. Get API key: https://dashscope.aliyun.com
2. Add to `.env`:
   ```
   QWEN_API_KEY=your_key_here
   QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

### Ollama (Fallback - Local)
1. Install: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull model: `ollama pull qwen2.5:7b-instruct`
3. Configure in `.env`:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:7b-instruct
   ```

## Testing LLM Integration

Run: `python manage.py shell`

```python
from apps.analysis.services.hybrid_llm_service import HybridLLMService

llm = HybridLLMService()

# Test OCR cleaning
cleaned, llm_used = llm.clean_ocr_text(
    raw_text="l. What is a data structure?",
    subject_name="Data Structures",
    year="2023"
)
print(f"Cleaned: {cleaned}")
print(f"LLM: {llm_used}")

# Test similarity
similar, conf, method, reason = llm.are_questions_similar(
    "Define stack",
    "What is a stack?"
)
print(f"Similar: {similar}, Confidence: {conf}, Method: {method}")
```

## Configuration Options

### OCR Enhancement Settings
In `config/settings.py`:

```python
OCR_ENHANCEMENT = {
    'USE_LLM_CLEANING': True,  # Enable LLM-based OCR cleaning
    'BATCH_PAGES': True,  # Clean multiple pages in one call
}
```

### Similarity Detection Settings
In `config/settings.py`:

```python
SIMILARITY_DETECTION = {
    'USE_HYBRID_APPROACH': True,  # Use embedding + LLM
    'USE_LLM_FOR_EDGE_CASES': True,  # LLM for uncertain cases
    'THRESHOLD_HIGH': 0.85,  # Definitely similar
    'THRESHOLD_LOW': 0.65,  # Definitely different
}
```

## Best Practices

1. **Use Gemini for OCR cleaning** - It's free and fast
2. **Enable batch processing** - Reduces API calls
3. **Set appropriate thresholds** - Balance accuracy vs API usage
4. **Monitor API usage** - Check statistics with:
   ```python
   llm = HybridLLMService()
   print(llm.stats)
   ```

## Troubleshooting

### Gemini API Errors
- Check API key is valid
- Verify you haven't exceeded daily quota
- Check network connectivity

### Ollama Connection Issues
- Ensure Ollama is running: `systemctl status ollama`
- Check base URL is correct
- Verify model is pulled: `ollama list`

### OCR Cleaning Not Working
- Check `OCR_ENHANCEMENT['USE_LLM_CLEANING']` is True
- Verify at least one LLM is configured
- Check logs for errors

### Similarity Detection Issues
- Review threshold settings
- Enable LLM for edge cases
- Check embedding service is working

## Performance Tips

1. **Batch OCR cleaning** - Process multiple pages together
2. **Cache results** - Avoid re-cleaning same text
3. **Use embeddings first** - Only call LLM for edge cases
4. **Monitor statistics** - Track which LLM is being used

## Security Considerations

1. **Never commit API keys** - Use environment variables
2. **Rate limit public uploads** - Already implemented (5/hour)
3. **Validate file uploads** - PDF validation enabled
4. **Monitor usage** - Track API costs and limits
