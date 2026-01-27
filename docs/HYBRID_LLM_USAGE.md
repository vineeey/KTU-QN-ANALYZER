# Hybrid LLM System Usage Guide

## Overview

The Hybrid LLM System enhances the KTU Question Analyzer with:
1. **OCR Text Cleaning**: AI-powered OCR error correction
2. **Smart Similarity Detection**: Two-tier hybrid approach

## Quick Start

### 1. Get Gemini API Key (FREE)
- Go to https://makersuite.google.com/app/apikey
- Sign in and create API key
- Free: 1,500 requests/day

### 2. Configure
Edit `.env`:
```bash
GEMINI_API_KEY=your_key_here
OCR_USE_LLM_CLEANING=true
SIMILARITY_USE_HYBRID=true
```

### 3. Install
```bash
pip install -r requirements.txt
```

## Features

### OCR Cleaning
Fixes: number confusion (l→1), question numbering, spacing, technical terms

### Similarity Detection
- Fast embedding check (99% cases)
- LLM verification for edge cases (1-5%)
- Thresholds: ≥0.85 similar, <0.65 different

## Usage Examples

```python
from apps.analysis.services.hybrid_llm_service import HybridLLMService

llm = HybridLLMService()

# Clean OCR
cleaned, used = llm.clean_ocr_text(raw_text, "Data Structures", "2023")

# Check similarity
similar, conf, method, reason = llm.are_questions_similar(q1, q2)
```

See full documentation for detailed usage, troubleshooting, and examples.
