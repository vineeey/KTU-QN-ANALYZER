# LLM Integration Improvements

## Overview

This document describes the improvements made to integrate LLM-based text cleaning and similarity detection throughout the KTU Question Analyzer pipeline.

## Problem Statement

The original implementation had the following issues:

1. **LLM not being used for text cleaning** - The HybridLLMService existed but was only being used in the OCR fallback path, not for regular PDF extraction
2. **OCR text had many mistakes** - Extracted text needed cleaning to fix common OCR errors
3. **Similar question identification not working properly** - The clustering service was only using embeddings without LLM verification
4. **Outputs not updating** - Re-processing papers resulted in duplicate/stale data

## Solutions Implemented

### 1. LLM Text Cleaning for All Extraction Methods

**Changes to `apps/analysis/pipeline.py`:**

- Added LLM text cleaning after **pdfplumber** extraction (lines 111-134)
- Added LLM text cleaning after **PyMuPDF** extraction (lines 130-158)
- Existing LLM cleaning for **OCR** fallback remains (lines 436-448)

**How it works:**
```python
# After extracting text with pdfplumber/PyMuPDF
if settings.OCR_ENHANCEMENT.get('USE_LLM_CLEANING', True):
    cleaned_text, llm_used = self.hybrid_llm.clean_ocr_text(
        raw_text=primary_text,
        subject_name=paper.subject.name,
        year=str(paper.year),
        use_advanced=True
    )
    if cleaned_text and llm_used != 'none':
        primary_text = cleaned_text
```

**Configuration:**
- Controlled by `OCR_ENHANCEMENT['USE_LLM_CLEANING']` setting
- Default: `true` (enabled by default)
- Can be disabled via environment variable: `OCR_USE_LLM_CLEANING=false`

### 2. Hybrid LLM Similarity Detection

**Changes to `apps/analytics/clustering.py`:**

- Added HybridLLMService initialization to TopicClusteringService (lines 40-50)
- Enhanced greedy clustering with hybrid LLM verification for edge cases (lines 207-228)
- Added post-clustering refinement using LLM to verify and split incorrectly merged clusters (lines 333-398)

**How it works:**

#### Edge Case Verification
```python
# During clustering, for similarity scores between low and high thresholds
if self.hybrid_llm and self.similarity_threshold <= similarity < self.same_question_threshold:
    is_similar, conf, method, reason = self.hybrid_llm.are_questions_similar(
        question.text, questions[j].text, marks1, marks2
    )
```

#### Post-Clustering Refinement
```python
# After initial clustering, verify each cluster
if self.hybrid_llm:
    clusters = self._refine_clusters_with_llm(clusters)
```

**Configuration:**
- Controlled by `SIMILARITY_DETECTION['USE_HYBRID_APPROACH']` setting
- Default: `true` (enabled by default)
- Can be disabled via environment variable: `SIMILARITY_USE_HYBRID=false`
- Uses threshold settings:
  - `THRESHOLD_HIGH`: 0.85 (definitely similar)
  - `THRESHOLD_LOW`: 0.65 (definitely different)
  - Between these values: LLM verification is used

### 3. Fixed Output Caching Issue

**Changes to `apps/analysis/pipeline.py`:**

- Added deletion of existing questions before re-processing a paper (lines 251-255)

**How it works:**
```python
# Delete existing questions for this paper to avoid duplicates/stale data
existing_question_count = Question.objects.filter(paper=paper).count()
if existing_question_count > 0:
    logger.info(f"Deleting {existing_question_count} existing questions for paper {paper.id}")
    Question.objects.filter(paper=paper).delete()
```

This ensures that:
- Re-processing a paper generates fresh analysis
- No duplicate questions are created
- Old/stale data is removed before new data is added

## Architecture

### LLM Service Hierarchy

1. **Primary LLM: Gemini 1.5 Flash**
   - Free tier: 1500 requests/day
   - Used for OCR cleaning and similarity verification
   - Configured via `GEMINI_API_KEY`

2. **Fallback LLM: Ollama (Local)**
   - Used when Gemini is unavailable or rate-limited
   - Configured via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`
   - Default model: `qwen2.5:7b-instruct`

3. **Embedding Model: Sentence Transformers**
   - Used for initial similarity scoring
   - Model: `multi-qa-MiniLM-L6-cos-v1`
   - Fast and accurate for question similarity

### Processing Flow

```
PDF Upload
    ↓
Extract Text (pdfplumber)
    ↓
Clean with LLM ← [NEW]
    ↓
Parse Questions
    ↓
Classify to Modules
    ↓
Create Question Records (delete old ones first) ← [NEW]
    ↓
Cluster Topics
    ↓
    ├─ Initial Clustering (embeddings)
    ├─ Edge Case Verification (LLM) ← [NEW]
    └─ Post-Clustering Refinement (LLM) ← [NEW]
    ↓
Generate Reports
```

## Configuration

### Environment Variables

```bash
# Enable/disable LLM text cleaning
OCR_USE_LLM_CLEANING=true

# Enable/disable hybrid similarity detection
SIMILARITY_USE_HYBRID=true

# Similarity thresholds
SIMILARITY_THRESHOLD_HIGH=0.85  # Definitely similar
SIMILARITY_THRESHOLD_LOW=0.65   # Definitely different

# Gemini API (Primary)
GEMINI_API_KEY=your_api_key_here

# Ollama (Fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

### Settings

All configuration is in `config/settings.py`:

```python
OCR_ENHANCEMENT = {
    'USE_LLM_CLEANING': os.environ.get('OCR_USE_LLM_CLEANING', 'true').lower() in ('true', '1', 'yes'),
    'USE_ADVANCED_PREPROCESSING': True,
    'BATCH_PAGES': os.environ.get('OCR_BATCH_PAGES', 'true').lower() in ('true', '1', 'yes'),
    'MAX_PAGES_PER_BATCH': 5,
}

SIMILARITY_DETECTION = {
    'USE_HYBRID_APPROACH': os.environ.get('SIMILARITY_USE_HYBRID', 'true').lower() in ('true', '1', 'yes'),
    'EMBEDDING_MODEL': os.environ.get('SIMILARITY_EMBEDDING_MODEL', 'multi-qa-MiniLM-L6-cos-v1'),
    'THRESHOLD_HIGH': float(os.environ.get('SIMILARITY_THRESHOLD_HIGH', '0.85')),
    'THRESHOLD_LOW': float(os.environ.get('SIMILARITY_THRESHOLD_LOW', '0.65')),
    'USE_LLM_FOR_EDGE_CASES': True,
}
```

## Testing

Run the test suite to verify LLM integration:

```bash
python test_llm_integration.py
```

This will test:
1. HybridLLMService initialization
2. OCR text cleaning functionality
3. Similar question identification
4. Clustering service integration

## Performance Considerations

### LLM Usage Optimization

1. **Text Cleaning**: LLM is only called once per extraction method per paper
2. **Similarity Detection**: 
   - Embedding comparison is done for all pairs (fast)
   - LLM verification only for edge cases (0.65-0.85 similarity range)
   - Post-clustering refinement only for multi-question clusters
3. **Caching**: Results are stored in database, re-analysis only on explicit re-processing

### Expected LLM Call Counts

For a typical paper with 20 questions:
- **Text Cleaning**: 1 LLM call (per extraction method)
- **Clustering**: 
  - Without LLM: 0 calls
  - With edge cases: 5-10 calls (approximately)
  - Post-refinement: 10-20 calls (approximately)
- **Total**: ~15-30 LLM calls per paper

With Gemini's free tier (1500 requests/day), you can process approximately **50-100 papers per day** with full LLM integration.

## Logging

The system logs LLM usage at various levels:

- **INFO**: Successful operations, LLM used, statistics
- **WARNING**: Fallback to alternative methods, LLM unavailable
- **ERROR**: LLM failures, extraction errors
- **DEBUG**: Detailed similarity checks, individual decisions

Example logs:
```
INFO: Applying LLM-based text cleaning...
INFO: ✓ Text cleaned using gemini
INFO: ✅ HybridLLMService initialized for similarity detection
DEBUG: Hybrid check Q5 vs Q7: True (0.92) via hybrid
INFO: LLM refinement split 3 questions into separate clusters
```

## Troubleshooting

### LLM Not Being Used

1. Check that `GEMINI_API_KEY` is set in `.env` file
2. Verify settings:
   - `OCR_ENHANCEMENT['USE_LLM_CLEANING']` should be `True`
   - `SIMILARITY_DETECTION['USE_HYBRID_APPROACH']` should be `True`
3. Check logs for initialization errors

### OCR Text Still Has Errors

1. LLM cleaning may not be perfect - it's a best-effort improvement
2. Try adjusting the OCR preprocessing settings
3. Consider using higher quality PDFs

### Similar Questions Not Being Grouped

1. Check similarity thresholds - they may be too high
2. Verify embedding model is loaded correctly
3. Enable DEBUG logging to see similarity scores
4. LLM refinement may be splitting overly broad clusters

### Outputs Not Updating

This should be fixed by the changes. If still occurring:
1. Check that questions are being deleted before re-processing
2. Verify clustering is being triggered after paper processing
3. Clear browser cache if viewing in web interface

## Future Improvements

Potential enhancements:

1. **Batch LLM Processing**: Process multiple questions in a single LLM call for efficiency
2. **Caching LLM Results**: Cache LLM responses for identical text to avoid duplicate calls
3. **Fine-tuned Models**: Train custom models on KTU question papers for better accuracy
4. **A/B Testing**: Compare embedding-only vs hybrid approach performance
5. **User Feedback Loop**: Allow users to correct similarity decisions to improve the system

## References

- HybridLLMService: `apps/analysis/services/hybrid_llm_service.py`
- Pipeline: `apps/analysis/pipeline.py`
- Clustering: `apps/analytics/clustering.py`
- Settings: `config/settings.py`
- Test Suite: `test_llm_integration.py`
