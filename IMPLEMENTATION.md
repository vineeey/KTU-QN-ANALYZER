# PYQ Analyzer - Implementation Summary

## Overview
A Django web application that analyzes Previous Year Question Papers (PYQs), identifies repeated topics, assigns priority levels, and generates comprehensive module-wise PDF reports for exam preparation.

## ✅ Implemented Features

### 1. Core Models
- **Subject**: Represents a subject with code, name, university, and exam board
- **Module**: Modules within a subject with topics and keywords
- **ExamPattern**: Configurable question-to-module mapping (KTU default included)
- **Paper**: Uploaded PDF question papers with processing status
- **Question**: Extracted questions with classification, embeddings, and analysis
- **TopicCluster**: Groups similar questions with priority tiers and repetition statistics
- **AnalysisJob**: Tracks background processing jobs

### 2. Question Extraction & Classification
- **PDF Text Extraction**: Uses pdfplumber and PyMuPDF for robust extraction
- **KTU Pattern Support**: Handles Part A (Q1-10, 3 marks) and Part B (Q11-20, 14 marks)
- **Module Classification**: 
  - Keyword-based matching (fast, no LLM required)
  - Exam pattern-based mapping (configurable per subject)
  - Optional hint extraction from PDF headers
- **Bloom's Taxonomy & Difficulty**: Rule-based classification

### 3. Topic Clustering & Repetition Analysis
- **Text Normalization**: Removes years, marks, trivial words for comparison
- **Similarity Matching**: Jaccard similarity for grouping similar questions
- **Priority Tiers**:
  - **Tier 1 (Top Priority)**: 4+ exam appearances - RED
  - **Tier 2 (High Priority)**: 3 exam appearances - ORANGE
  - **Tier 3 (Medium Priority)**: 2 exam appearances - YELLOW
  - **Tier 4 (Low Priority)**: 1 exam appearance - GRAY
- **Frequency Tracking**: Years appeared, total marks, Part A/B distribution

### 4. Module-Wise PDF Reports
Generated PDFs include:
- **Header**: Module name, subject details, university, scheme
- **Part A Section**: Questions grouped by year with marks
- **Part B Section**: Questions grouped by year with sub-parts
- **Repeated Question Analysis**: Topics grouped by priority tier
- **Final Study Priority Order**: Ranked table with study recommendations
- **Study Strategy Guide**: Time allocation suggestions

### 5. Analytics Dashboard
- **Subject-Level Dashboard**:
  - Overview statistics (papers, questions, topics)
  - Top 3 topics per module visualization
  - Module distribution charts
- **Module-Level Analytics**:
  - Topic frequency bar charts (Chart.js)
  - Priority tier breakdown
  - Detailed topic listing with years and marks
  - Color-coded priority badges

### 6. Background Task Processing
- **Paper Analysis**: Asynchronous extraction and classification
- **Topic Clustering**: Background job for repetition analysis
- **Django-Q2**: SQLite-based task queue (no Redis/Celery needed)

### 7. API Endpoints
- `/analytics/subject/<id>/api/`: JSON data for Chart.js graphs
- Topic frequency data per module
- Module distribution statistics

## 🗂️ Application Structure

```
apps/
├── subjects/         # Subject and Module management
│   ├── models.py     # Subject, Module, ExamPattern
│   └── views.py      # CRUD views for subjects
├── papers/           # PDF upload and management
│   ├── models.py     # Paper, PaperPage
│   └── views.py      # Batch upload with deduplication
├── questions/        # Question storage
│   ├── models.py     # Question with embeddings
│   └── views.py      # Question listing and editing
├── analysis/         # Analysis pipeline
│   ├── pipeline.py   # Orchestrates extraction → classification
│   ├── tasks.py      # Background tasks
│   └── services/     # Extractors, classifiers, embedders
├── analytics/        # Topic analysis and statistics
│   ├── models.py     # TopicCluster
│   ├── clustering.py # Topic grouping logic
│   ├── calculator.py # Stats computation
│   └── views.py      # Dashboard views
└── reports/          # PDF report generation
    ├── module_report_generator.py  # Enhanced generator
    └── views.py      # Download endpoints

templates/
├── analytics/
│   ├── dashboard.html       # Main analytics view
│   └── module_detail.html   # Module-specific analytics
├── reports/
│   ├── reports_list_new.html       # Report downloads
│   └── module_report_detailed.html # PDF template
└── papers/
    └── paper_upload.html    # Multi-file upload
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Migrations
```bash
python manage.py migrate
```

### 3. Create Test Data
```bash
python manage.py setup_test_data
```

This creates:
- Admin user (admin@test.com / admin123)
- Sample subject: Disaster Management (MCN301)
- 5 modules with topics and keywords
- KTU exam pattern configuration

### 4. Start Django-Q Worker (for background tasks)
```bash
python manage.py qcluster
```

### 5. Run Development Server
```bash
python manage.py runserver
```

## 📋 User Flow

### Step 1: Create/Select Subject
- Navigate to Subjects
- Create new subject or select existing
- Configure modules and exam pattern

### Step 2: Upload Question Papers
- Go to Papers → Upload
- Select multiple PDF files (batch upload supported)
- Files are automatically processed in background
- Deduplication by SHA-256 hash

### Step 3: Trigger Topic Analysis
- Once papers are processed, go to Analytics
- Click "Analyze Topics" button
- Wait for clustering to complete (runs in background)

### Step 4: View Analytics
- **Subject Dashboard**: Overview and top topics
- **Module Analytics**: Detailed frequency charts
- Interactive Chart.js visualizations

### Step 5: Download Reports
- Go to Reports section
- Download individual module PDFs
- Or download all modules at once
- Each PDF contains:
  - All questions by year
  - Priority rankings
  - Study recommendations

## 🔧 Configuration

### Exam Pattern
Edit `ExamPattern` in Django admin or database:
```python
pattern_config = {
    'part_a': {'1': 1, '2': 1, '3': 2, ...},  # Q_num → Module
    'part_b': {'11': 1, '12': 1, '13': 2, ...}
}
```

### Priority Thresholds
Edit in code or make configurable:
```python
tier_1_threshold = 4  # Top Priority
tier_2_threshold = 3  # High Priority
tier_3_threshold = 2  # Medium Priority
```

### Similarity Threshold
Adjust in `TopicClusteringService`:
```python
similarity_threshold = 0.75  # 75% similarity to group
```

## 📊 Key Algorithms

### Question Extraction
1. Extract text from PDF using pdfplumber/PyMuPDF
2. Clean and normalize (remove headers, footers)
3. Identify Part A (short answer) and Part B (long answer)
4. Parse question numbers, marks, sub-parts
5. Handle OCR artifacts and common errors

### Module Classification
1. **Pattern-based**: Use ExamPattern config (Q1-2 → Module 1)
2. **Keyword matching**: Score questions against module keywords
3. **Fallback**: Assign to Module 1 if unmatched

### Topic Clustering
1. Normalize question text (remove years, marks, trivial words)
2. Compute Jaccard similarity between questions
3. Group questions with similarity > threshold
4. Extract topic name from representative question
5. Calculate statistics (frequency, years, marks)
6. Assign priority tier based on frequency

## 🎨 Frontend Features

### Technologies
- **Tailwind CSS**: Responsive styling
- **Chart.js**: Interactive graphs
- **Lucide Icons**: Icon library
- **Alpine.js**: Lightweight interactivity (if needed)

### Color Scheme
- Tier 1 (Top): Red (`#ef4444`)
- Tier 2 (High): Orange (`#f97316`)
- Tier 3 (Medium): Yellow (`#eab308`)
- Tier 4 (Low): Gray (`#9ca3af`)

### Responsive Design
- Mobile-first approach
- Grid layouts for cards
- Collapsible sections
- Touch-friendly buttons

## ⚠️ Known Limitations

1. **PDF Quality**: Scanned/low-quality PDFs may have extraction errors
2. **OCR**: Not fully implemented for image-based PDFs
3. **Similarity**: Text-based matching (embeddings available but not fully integrated)
4. **Multi-language**: Currently English only
5. **Bulk Downloads**: Individual module PDFs (ZIP not implemented)

## 🔜 Future Enhancements

1. **Enhanced Similarity**: Use embeddings (SentenceTransformers) for better clustering
2. **Configuration UI**: Web interface for exam patterns and thresholds
3. **Multi-university Support**: Templates for different exam boards
4. **OCR Integration**: Tesseract for scanned PDFs
5. **Question Tagging**: Manual topic assignment by users
6. **Export Formats**: Excel, CSV, JSON exports
7. **Mobile App**: React Native companion app
8. **Collaborative Features**: Share subjects/reports between users

## 📝 Testing

### Manual Testing Checklist
- [ ] Upload single PDF
- [ ] Upload multiple PDFs (batch)
- [ ] Check question extraction accuracy
- [ ] Verify module classification
- [ ] Run topic analysis
- [ ] View analytics dashboard
- [ ] Download module PDF
- [ ] Check PDF formatting
- [ ] Test with different subjects

### Unit Tests (To Be Added)
```bash
pytest apps/analysis/tests/
pytest apps/analytics/tests/
```

## 📚 Documentation

- **Models**: See docstrings in `models.py` files
- **Services**: See docstrings in `services/` directories
- **Views**: See docstrings in `views.py` files
- **Templates**: Comments in HTML files

## 🤝 Contributing

1. Follow Django best practices
2. Write descriptive commit messages
3. Add docstrings to all functions
4. Test on low-spec hardware (HP 15s with Ryzen 3)
5. Keep dependencies minimal

## 📄 License

[Add your license here]

## 👥 Credits

Developed for analyzing KTU (APJ Abdul Kalam Technological University) question papers,
but adaptable to any university/exam board.

---

**Note**: This is a zero-cost solution running entirely on SQLite with no external APIs or paid services.
Designed to work on modest hardware (8GB RAM, CPU-only).
