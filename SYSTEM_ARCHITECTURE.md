# PYQ Analyzer - Complete System Architecture

## Overview
Django web application for analyzing Previous Year Question (PYQ) papers, extracting questions, clustering topics by repetition frequency, assigning priority levels, and generating module-wise PDF reports with analytics.

---

## Core Features Implemented

### 1. **Configurable Exam Patterns** ✅
- **Model**: `ExamPattern` in `apps/subjects/models.py`
- **Purpose**: Maps question numbers to modules (not hardcoded)
- **Configuration**: Stored as JSON per subject
- **Example KTU Pattern**:
  ```json
  {
    "part_a": {"1": 1, "2": 1, "3": 2, "4": 2, "5": 3, "6": 3, "7": 4, "8": 4, "9": 5, "10": 5},
    "part_b": {"11": 1, "12": 1, "13": 2, "14": 2, "15": 3, "16": 3, "17": 4, "18": 4, "19": 5, "20": 5}
  }
  ```
- **Usage**: Automatically maps questions to modules during extraction
- **Admin Interface**: Configurable via Django admin at `/admin/subjects/exampattern/`

---

### 2. **Hybrid Classification System** ✅

#### **For KTU Papers (Pattern-Based)**
Location: `apps/analysis/pipeline.py` (lines 93-103)
```python
# Check exam pattern first
if hasattr(subject, 'exam_pattern'):
    exam_pattern = subject.exam_pattern
    module_num = exam_pattern.get_module_for_question(
        question_number, part
    )
```

#### **For Non-KTU Papers (AI-Based)**
Location: `apps/analysis/services/classifier.py`

**3-Tier Classification Strategy**:

1. **Module Hints from PDF** (Highest Priority)
   - Extracts "Module 1", "Module 2" headers from PDFs
   - Direct assignment if detected

2. **Keyword Matching** (Fast, No API calls)
   ```python
   def classify_by_keywords(self, question_text, modules):
       """
       Enhanced keyword matching with:
       - Module keywords (custom per module)
       - Module topics
       - Module name words
       - Default keywords for common subjects
       """
   ```
   
   Default keywords for Disaster Management:
   - Module 1: disaster, hazard, vulnerability, classification
   - Module 2: mitigation, preparedness, prevention, DRR
   - Module 3: NDMA, SDMA, policy, institutional framework
   - Module 4: response, relief, rehabilitation, emergency
   - Module 5: community, participation, CBDM, awareness

3. **LLM Classification** (Fallback, Optional)
   ```python
   def _classify_with_llm(self, question_text, subject, modules):
       """Uses Ollama LLM to classify difficult questions"""
   ```
   - Only used if keyword matching fails
   - Configurable LLM client (Ollama)
   - Batch processing for efficiency

---

### 3. **PDF Extraction Pipeline** ✅

#### **Supported Libraries**:
- **pdfplumber** (Primary) - Text-based PDFs
- **PyPDF2** (Fallback) - Alternative text extraction
- **PyMuPDF** (Optional) - High-performance extraction
- **OCR**: Tesseract support for scanned PDFs (future)

#### **Question Extraction Process**:
Location: `apps/analysis/views.py` - `_extract_ktu_questions_improved()`

**Multi-Pattern Regex Matching**:
```python
patterns = [
    r'(Q\d+[a-z]?)\.\s*(.+?)(?=Q\d+|$)',           # Q1. Question text
    r'(\d+)\.\s*(.+?)(?=\d+\.|$)',                   # 1. Question text
    r'Question\s+(\d+[a-z]?)\s*[:\.]?\s*(.+?)(?=Question|\Z)',  # Question 1: text
]
```

**Extracts**:
- Question number
- Question text
- Marks (3, 14, 8, etc.)
- Part (A or B)
- Module hints from headers
- Year/month from title

---

### 4. **Topic Clustering & Repetition Analysis** ✅

Location: `apps/analytics/clustering.py`

#### **Clustering Algorithm**:
```python
class TopicClusteringService:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        tier_1_threshold: int = 4,
        tier_2_threshold: int = 3,
        tier_3_threshold: int = 2
    ):
```

**Process**:
1. **Text Normalization**:
   ```python
   def _normalize_text(self, text):
       # Remove marks, years, question numbers
       # Remove trivial words
       # Lowercase and clean whitespace
   ```

2. **Similarity Calculation** (Jaccard Index):
   ```python
   def _calculate_text_similarity(self, text1, text2):
       tokens1 = set(text1.split())
       tokens2 = set(text2.split())
       return len(intersection) / len(union)
   ```

3. **Cluster Creation**:
   - Groups similar questions (similarity >= threshold)
   - Generates human-readable topic names
   - Tracks frequency across years
   - Assigns priority tiers

4. **Priority Tier Assignment**:
   - **Tier 1 (TOP)**: 4+ appearances
   - **Tier 2 (HIGH)**: 3 appearances
   - **Tier 3 (MEDIUM)**: 2 appearances
   - **Tier 4 (LOW)**: 1 appearance

---

### 5. **Database Schema** ✅

#### **Core Models**:

**Subject** (`apps/subjects/models.py`):
```python
class Subject(SoftDeleteModel):
    name = CharField(max_length=255)
    code = CharField(max_length=50)
    scheme = CharField(max_length=100)
    university = CharField(max_length=255)
    user = ForeignKey(User)
```

**Module** (`apps/subjects/models.py`):
```python
class Module(SoftDeleteModel):
    subject = ForeignKey(Subject)
    number = PositiveIntegerField()
    name = CharField(max_length=255)
    topics = JSONField(default=list)
    keywords = JSONField(default=list)
```

**ExamPattern** (`apps/subjects/models.py`):
```python
class ExamPattern(SoftDeleteModel):
    subject = OneToOneField(Subject)
    name = CharField(max_length=255)
    pattern_config = JSONField()  # Question → Module mapping
    part_a_marks = PositiveIntegerField(default=3)
    part_b_marks = PositiveIntegerField(default=14)
```

**Paper** (`apps/papers/models.py`):
```python
class Paper(SoftDeleteModel):
    subject = ForeignKey(Subject)
    file = FileField(upload_to='papers/')
    title = CharField(max_length=500)
    year = PositiveIntegerField()
    month = CharField(max_length=50)
    status = CharField(choices=ProcessingStatus.choices)
```

**Question** (`apps/questions/models.py`):
```python
class Question(BaseModel):
    paper = ForeignKey(Paper)
    module = ForeignKey(Module, null=True)
    question_number = CharField(max_length=10)
    text = TextField()
    marks = PositiveIntegerField(null=True)
    part = CharField(max_length=10)  # A or B
    embedding = ArrayField(null=True)
    topic_cluster = ForeignKey(TopicCluster, null=True)
```

**TopicCluster** (`apps/analytics/models.py`):
```python
class TopicCluster(BaseModel):
    subject = ForeignKey(Subject)
    module = ForeignKey(Module, null=True)
    topic_name = CharField(max_length=500)
    normalized_key = CharField(max_length=500)
    frequency_count = PositiveIntegerField()
    years_appeared = JSONField(default=list)
    total_marks = PositiveIntegerField(default=0)
    priority_tier = CharField(choices=PriorityTier.choices)
```

---

### 6. **PDF Report Generation** ✅

Location: `apps/reports/ktu_report_generator.py`

#### **Technology**: ReportLab (Python PDF library)

#### **Report Structure** (Per Module):
1. **Title Section**:
   - Module number and subject name
   - KTU scheme year

2. **PART A Section** (3 marks each):
   ```
   December 2021
   • Question text — (Dec 2021, 3 marks)
   • Question text — (Dec 2021, 3 marks)
   
   December 2022
   • Question text — (Dec 2022, 3 marks)
   ```

3. **PART B Section** (14 marks each):
   ```
   December 2021
   Qn 11
   • Sub-question — (Dec 2021, 14 marks)
   Qn 12
   • Sub-question — (Dec 2021, 14 marks)
   ```

4. **Repeated Question Analysis**:
   ```
   TOP PRIORITY — Repeated 4+ Times
   1. Layers of atmosphere
   Appears in: 2021, 2022, 2024
   
   HIGH PRIORITY — Repeated 3 Times
   2. Indian monsoon system
   Appears in: 2021, 2023, 2024
   ```

5. **Final Study Priority Order**:
   ```
   Tier 1 (Most repeated — must learn first)
   • Layers of atmosphere
   • Greenhouse effect
   
   Tier 2 (Frequently repeated)
   • Indian monsoon system
   ```

#### **Features**:
- Automatic year grouping
- Question number formatting
- Priority tier coloring
- Page breaks between sections
- Downloadable as single PDF per module
- Downloadable as ZIP for all modules

---

### 7. **Analytics Dashboard** ✅

Location: `apps/analytics/views.py`, `templates/analytics/`

#### **Visualizations**:
1. **Module-wise Question Distribution** (Bar Chart)
2. **Topic Frequency Heatmap**
3. **Priority Tier Distribution** (Pie Chart)
4. **Year-wise Analysis** (Line Chart)
5. **Part A vs Part B Distribution**

#### **Technology**: Chart.js (JavaScript charting library)

---

## User Flow

### Complete Workflow:

1. **Create Subject**:
   ```
   Go to: /subjects/create/
   Input: Name, Code, Scheme, University
   Select: Exam Pattern (KTU or create custom)
   ```

2. **Upload Papers**:
   ```
   Go to: /subjects/{subject_id}/
   Upload: Multiple PDF files
   System: Extracts questions automatically
   ```

3. **Analysis**:
   ```
   Click: "Analyze Papers" button
   System: 
   - Extracts all questions
   - Classifies into modules (pattern or AI)
   - Clusters similar questions
   - Calculates repetition frequency
   - Assigns priority tiers
   ```

4. **View Analytics**:
   ```
   Go to: /analytics/subject/{subject_id}/
   View: Interactive graphs and statistics
   ```

5. **Download Reports**:
   ```
   Go to: /reports/subject/{subject_id}/
   Download: Individual module PDFs or ZIP of all
   ```

---

## Configuration Options

### Similarity Threshold
Default: 0.60 (60% similarity)
Location: `apps/analysis/views.py` line 116
```python
cluster_stats = analyze_subject_topics(subject, similarity_threshold=0.60)
```

### Priority Tier Thresholds
Configurable in: `TopicClusteringService.__init__()`
```python
tier_1_threshold: int = 4  # Top Priority
tier_2_threshold: int = 3  # High Priority
tier_3_threshold: int = 2  # Medium Priority
# tier_4 = 1 (Low Priority)
```

### LLM Configuration
Location: `services/llm/ollama_client.py`
```python
class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2"
    ):
```

---

## API Endpoints

### Subjects
- `GET /subjects/` - List all subjects
- `POST /subjects/create/` - Create new subject
- `GET /subjects/{id}/` - Subject detail
- `POST /subjects/{id}/upload/` - Upload papers

### Analysis
- `POST /analysis/subject/{id}/analyze/` - Analyze papers
- `POST /analysis/subject/{id}/reset/` - Reset & re-analyze

### Analytics
- `GET /analytics/subject/{id}/` - Dashboard
- `GET /analytics/subject/{id}/module/{number}/` - Module detail

### Reports
- `GET /reports/subject/{id}/` - Reports list
- `GET /reports/subject/{id}/module/{number}/` - Download module PDF
- `GET /reports/subject/{id}/all/` - Download all modules (ZIP)

---

## Technology Stack

### Backend
- **Django 5.2.9** - Web framework
- **Python 3.14** - Programming language
- **SQLite3** - Database
- **Celery** (Optional) - Background task queue

### PDF Processing
- **pdfplumber** - Text extraction
- **PyPDF2** - Fallback extraction
- **ReportLab** - PDF generation

### Machine Learning
- **Ollama** - Local LLM inference (optional)
- **SentenceTransformers** (Future) - Embeddings
- **scikit-learn** - Clustering algorithms

### Frontend
- **Tailwind CSS** - Styling
- **Chart.js** - Data visualization
- **Alpine.js** - Interactive components
- **Lucide Icons** - Icon library

---

## Performance Optimizations

1. **Batch Processing**: Questions classified in batches for efficiency
2. **Keyword Matching First**: Avoids expensive LLM calls when possible
3. **Database Indexing**: Optimized queries with proper indexes
4. **Lazy Loading**: Questions loaded on-demand
5. **Caching**: Report data cached after first generation

---

## Future Enhancements

1. **OCR Support**: Tesseract integration for scanned PDFs
2. **Advanced Embeddings**: SentenceTransformers for better clustering
3. **Export Options**: Excel, Word, JSON exports
4. **Mobile App**: React Native mobile interface
5. **Collaborative Features**: Share subjects with other users
6. **AI-Generated Summaries**: LLM-based topic summaries
7. **Spaced Repetition**: Study scheduler based on priority

---

## Current Status: FULLY FUNCTIONAL ✅

All core features are implemented and working:
- ✅ Configurable exam patterns (not hardcoded)
- ✅ AI-based classification for non-KTU papers
- ✅ Topic clustering with configurable thresholds
- ✅ Priority tier assignment
- ✅ Module-wise PDF generation
- ✅ Analytics dashboard with graphs
- ✅ SQLite3 database with all metadata
- ✅ Pattern-based + AI hybrid classification

## Access the Application
- Server: `http://127.0.0.1:8000/`
- Admin: `http://127.0.0.1:8000/admin/`
- Subjects: `http://127.0.0.1:8000/subjects/`
