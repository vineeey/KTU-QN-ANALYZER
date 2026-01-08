# KTU Previous Year Question Priority Analyzer

**REFACTORED ARCHITECTURE - FOLLOWS SPECIFICATION EXACTLY**

## 🎯 Core Goal

This system allows **WEB USERS (NO LOGIN)** to upload multiple KTU previous-year question paper PDFs and automatically generate **MODULE-WISE PDFs** containing:

1. **Complete Question Bank** (all extracted questions, module-wise)
2. **Priority Analysis** that ranks REPEATED TOPICS based on:
   - Frequency across years
   - Marks weight  
   - Combined Part A + Part B contribution
   - Confidence score (year coverage)

**Priority classification is the CORE feature.**

---

## 🏗️ Architecture Overview

### Design Principles

✅ **NO permanent storage** - All data is job-scoped and temporary  
✅ **NO login required** - Pure guest workflow  
✅ **Job-based isolation** - Each upload session gets a UUID  
✅ **Auto-cleanup** - Data deleted after 24 hours  
✅ **Deterministic rules** - AI only for semantic similarity  
✅ **Classical NLP** - No OpenAI, no GPT, no paid APIs

### Tech Stack

**Backend:**
- Django (job orchestration)
- Python (pipeline logic)

**PDF Processing:**
- pdfplumber (text-based PDFs)
- PyMuPDF + pytesseract + OpenCV (scanned PDFs)

**NLP/ML:**
- sentence-transformers/all-MiniLM-L6-v2 (embeddings)
- HDBSCAN (clustering)
- scikit-learn
- NumPy

**PDF Generation:**
- ReportLab / WeasyPrint

**Frontend:**
- Django templates
- Vanilla JavaScript
- CSS (no framework)

---

## 📁 Project Structure

```
apps/
├── analysis/                    # Core pipeline orchestration
│   ├── models.py               # AnalysisJob (UUID-based, temporary)
│   ├── job_models.py           # TempPaper, TempQuestion, TempTopicCluster
│   ├── pipeline_13phases.py    # Complete 13-phase implementation
│   ├── services/               # Isolated service modules
│   │   ├── pdf_extractor.py   # Phase 2: PDF detection
│   │   ├── extractor.py        # Phase 3: Question segmentation
│   │   ├── embedder.py         # Phase 6: Embedding generation
│   │   └── clustering.py       # Phase 7: Topic clustering
│   └── management/commands/
│       └── cleanup_expired_jobs.py  # Phase 13: Auto-cleanup
│
├── core/
│   └── guest_views.py          # NO-LOGIN upload & download views
│
├── reports/
│   └── ktu_report_generator_new.py  # Phase 11: Module PDF generation
│
└── [users/, subjects/, papers/, etc.]  # OPTIONAL authenticated features

templates/
├── pages/
│   └── guest_upload.html       # Landing page with upload form
└── analysis/
    └── job_results.html        # Download page with module links

config/
├── settings.py                 # Django configuration
└── urls.py                     # Root = guest upload (NO login)
```

---

## 🔄 13-Phase Workflow (EXACT IMPLEMENTATION)

### PHASE 1: User Upload
```python
class Phase1_Upload:
    - Accept multiple PDFs
    - Validate file type/size
    - Generate job_id (UUID)
    - Create /media/jobs/<job_id>/
```

### PHASE 2: PDF Type Detection
```python
class Phase2_PDFDetection:
    - Detect text-based vs scanned
    - Extract raw text (UNCHANGED)
    - Store separately
```

### PHASE 3: Question Segmentation (RULE-BASED)
```python
class Phase3_QuestionSegmentation:
    - Detect PART A and PART B
    - Extract: number, text, marks, year, part
    - Handle OR questions
    - Each logical question = one unit
```

### PHASE 4: Module Mapping (RULE-BASED)
```python
class Phase4_ModuleMapping:
    KTU_MODULE_MAPPING = {
        1: 1, 2: 1, 11: 1, 12: 1,   # Module 1
        3: 2, 4: 2, 13: 2, 14: 2,   # Module 2
        5: 3, 6: 3, 15: 3, 16: 3,   # Module 3
        7: 4, 8: 4, 17: 4, 18: 4,   # Module 4
        9: 5, 10: 5, 19: 5, 20: 5,  # Module 5
    }
```

### PHASE 5: Question Normalization
```python
class Phase5_Normalization:
    - Create normalized_text field
    - Remove numbering, marks, year refs
    - Preserve academic meaning
    - DO NOT overwrite raw_text
```

### PHASE 6: Embedding Generation
```python
class Phase6_Embeddings:
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - 384-dimensional vectors
    - Module-wise processing
    - Batch processing for efficiency
```

### PHASE 7: Topic Clustering (CORE AI)
```python
class Phase7_Clustering:
    - Algorithm: HDBSCAN
    - Per-module clustering
    - Each cluster = one exam topic
    - Noise questions allowed (cluster_id=-1)
```

### PHASE 8: Priority Scoring (CORE FEATURE)
```python
class Phase8_PriorityScoring:
    def calculate_metrics(self, total_years):
        # Frequency (distinct years)
        self.frequency = len(distinct_years)
        
        # Marks statistics
        self.average_marks = total_marks / question_count
        
        # 🔥 NEW: Part A vs Part B contribution
        self.part_a_count = questions.filter(part='A').count()
        self.part_b_count = questions.filter(part='B').count()
        
        # 🔥 NEW: Confidence score
        self.confidence_score = (frequency / total_years) * 100
        
        # Priority score
        self.priority_score = (2 × frequency) + average_marks
```

### PHASE 9: Confidence Score (MANDATORY EXTRA FEATURE)
```
Confidence (%) = (Years appeared ÷ Total years) × 100

Example:
- Topic appeared in 6 out of 7 years → 85.7% confidence
```

### PHASE 10: Priority Tiers
```python
TIER_1 = Frequency >= 4  # Very High Priority
TIER_2 = Frequency >= 3  # High Priority  
TIER_3 = Frequency >= 2  # Medium Priority
TIER_4 = Frequency >= 1  # Low Priority
```

### PHASE 11: Module-wise PDF Generation
```python
class Phase11_PDFGeneration:
    Each PDF contains:
    
    SECTION A — COMPLETE QUESTION BANK
    - Part A (year-wise grouping)
    - Part B (year-wise grouping)
    
    SECTION B — REPEATED QUESTION ANALYSIS
    - Tier-wise topics
    - Repetition count
    - Appearing years
    - 🔥 Confidence score
    - 🔥 Part A vs Part B contribution
    
    FINAL STUDY PRIORITY ORDER
    - Linear list from Tier 1 → Tier 4
```

### PHASE 12: User Delivery
```
Show download buttons for Module 1-5 PDFs
```

### PHASE 13: Auto Cleanup
```python
class Phase13_Cleanup:
    def cleanup_job(job):
        - Delete workspace directory
        - Delete all temp models
        - Job cascade-deletes all related data
    
    # Cron task:
    python manage.py cleanup_expired_jobs
```

---

## 🚀 User Flow (NO LOGIN REQUIRED)

```
1. User opens website (/)
   ↓
2. User uploads MULTIPLE PYQ PDFs (same subject)
   ↓
3. System creates job_id (UUID) and workspace
   ↓
4. System runs 13-phase pipeline (2-5 minutes)
   ↓
5. System generates 5 module-wise PDFs
   ↓
6. User sees download page with Module 1-5 buttons
   ↓
7. User downloads PDFs
   ↓
8. System auto-cleans after 24 hours
```

**NO login, NO registration, NO permanent storage.**

---

## 📊 Data Models (Job-Scoped, Temporary)

### AnalysisJob (UUID primary key)
```python
class AnalysisJob(models.Model):
    id = UUIDField(primary_key=True)
    subject_name = CharField()  # No user FK!
    status = CharField()        # Created → Processing → Completed
    total_years = IntegerField()
    years_list = JSONField()
    output_pdfs = JSONField()   # Module paths
    expires_at = DateTimeField()
```

### TempPaper (Job-scoped)
```python
class TempPaper(models.Model):
    id = UUIDField()
    job = ForeignKey(AnalysisJob, on_delete=CASCADE)
    file = FileField()
    year = CharField()
    pdf_type = CharField()      # text/scanned/hybrid
    raw_text = TextField()      # UNCHANGED
```

### TempQuestion (Job-scoped)
```python
class TempQuestion(models.Model):
    id = UUIDField()
    paper = ForeignKey(TempPaper, on_delete=CASCADE)
    question_number = CharField()
    part = CharField()          # A or B
    marks = IntegerField()
    raw_text = TextField()      # Original
    normalized_text = TextField()  # For embeddings
    module_number = IntegerField()  # 1-5
    embedding = JSONField()     # 384-dim vector
    topic_cluster = ForeignKey(TempTopicCluster)
```

### TempTopicCluster (Job-scoped)
```python
class TempTopicCluster(models.Model):
    id = UUIDField()
    job = ForeignKey(AnalysisJob, on_delete=CASCADE)
    module_number = IntegerField()
    cluster_id = IntegerField()
    topic_label = CharField()
    
    # PHASE 8 metrics
    frequency = IntegerField()          # Distinct years
    years_appeared = JSONField()
    average_marks = FloatField()
    
    # 🔥 NEW FEATURES
    part_a_count = IntegerField()       # Part A appearances
    part_b_count = IntegerField()       # Part B appearances
    confidence_score = FloatField()     # Percentage
    
    # Priority
    priority_score = FloatField()       # (2×freq) + avg_marks
    priority_tier = IntegerField()      # 1-4
```

---

## 🖨️ PDF Output Format

### Example: Module 1 PDF

```
═══════════════════════════════════════════════════════════════
MODULE 1 - PRIORITY ANALYSIS REPORT
═══════════════════════════════════════════════════════════════
Subject: Disaster Management
Analysis Date: 2026-01-08
Years Analyzed: 2021, 2022, 2023, 2024, 2025 (5 years)

───────────────────────────────────────────────────────────────
SECTION A: COMPLETE QUESTION BANK
───────────────────────────────────────────────────────────────

📌 PART A (3 marks each)

2025:
  1. Define disaster risk reduction. (3)
  2. Explain vulnerability in disaster context. (3)

2024:
  1. What is hazard? Give examples. (3)
  2. List components of disaster management cycle. (3)

[... more years ...]

📌 PART B (14 marks each)

2025:
  11. Discuss disaster risk management framework in detail. (14)
  12. Explain structural and non-structural mitigation. (14)

[... more years ...]

───────────────────────────────────────────────────────────────
SECTION B: REPEATED QUESTION ANALYSIS
───────────────────────────────────────────────────────────────

🔥 TIER 1 - VERY HIGH PRIORITY

1. Disaster Risk Management (Framework + Core Elements)
   Appears in: 2021, 2022, 2023, 2024, 2025
   Repetition count: 6
   Confidence: 85%
   
   Appears as:
   • Part A: 2 times
   • Part B: 4 times
   
   → Very high probability long-answer topic.

2. Mitigation Strategies (Structural vs Non-Structural)
   Appears in: 2021, 2023, 2024, 2025
   Repetition count: 5
   Confidence: 80%
   
   Appears as:
   • Part A: 1 time
   • Part B: 4 times

⚡ TIER 2 - HIGH PRIORITY

3. Vulnerability Assessment
   Appears in: 2022, 2023, 2024
   Repetition count: 3
   Confidence: 60%
   
   Appears as:
   • Part A: 2 times
   • Part B: 1 time
   
   → More frequent in short-answer format.

[... more topics ...]

───────────────────────────────────────────────────────────────
FINAL STUDY PRIORITY ORDER
───────────────────────────────────────────────────────────────

✅ TIER 1 (Must Study)
  1. Disaster Risk Management (Framework)
  2. Mitigation Strategies

✅ TIER 2 (High Priority)
  3. Vulnerability Assessment
  4. Disaster Management Cycle

✅ TIER 3 (Medium Priority)
  5. Early Warning Systems
  6. Community Preparedness

✅ TIER 4 (Low Priority)
  7. Case Studies
  8. Specific Examples

═══════════════════════════════════════════════════════════════
```

---

## 🔒 No Login System

**Authentication is OPTIONAL, not required.**

- Root URL (`/`) serves guest upload page
- All core features work WITHOUT login
- `job_id` serves as access token
- Users can optionally register to save history (future feature)

**Current implementation:**
```python
# config/urls.py
urlpatterns = [
    path('', include('apps.core.guest_views')),  # NO LOGIN
    path('users/', ...),  # OPTIONAL
]
```

---

## 🧹 Auto-Cleanup Mechanism

### Manual Cleanup
```bash
python manage.py cleanup_expired_jobs
```

### Automated Cleanup (Cron)
```bash
# Add to crontab
0 * * * * cd /path/to/project && python manage.py cleanup_expired_jobs
```

### Django-Q Scheduled Task
```python
from django_q.tasks import schedule
from django_q.models import Schedule

schedule(
    'apps.analysis.management.commands.cleanup_expired_jobs.Command',
    schedule_type=Schedule.HOURLY
)
```

### Expiry Rules
- **Completed jobs:** 24 hours after completion
- **Failed jobs:** 1 hour after failure
- **Abandoned jobs:** 48 hours after creation

---

## 🚦 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Sentence Transformer Model
```bash
python scripts/download_models.py
```

### 3. Run Migrations
```bash
python manage.py makemigrations analysis
python manage.py migrate
```

### 4. Start Development Server
```bash
python manage.py runserver
```

### 5. Access Guest Upload
```
http://localhost:8000/
```

### 6. Upload PDFs
- Enter subject name
- Upload 5-10 previous year PDFs
- Wait for processing
- Download module PDFs

---

## 📈 Confidence Score Feature

**Formula:**
```
Confidence (%) = (Years topic appeared ÷ Total years uploaded) × 100
```

**Interpretation:**
- **85-100%** - Almost guaranteed to appear
- **60-84%** - High probability
- **40-59%** - Medium probability  
- **0-39%** - Low probability

**Example:**
```
Topic: "Disaster Risk Framework"
Appeared in: 2021, 2022, 2023, 2024, 2025
Total years uploaded: 7
Confidence = (5 ÷ 7) × 100 = 71.4%

Interpretation: High probability topic
```

---

## 🔀 Part A vs Part B Analysis

**Insight:** Understanding question format helps exam preparation.

**Example:**
```
Topic: "Vulnerability Assessment"
Part A: 4 times (short notes)
Part B: 1 time (long answer)

→ Usually asked as short-answer format
→ Focus on concise definitions
```

**Contrast:**
```
Topic: "Mitigation Strategies"
Part A: 1 time
Part B: 6 times

→ Deep topic requiring detailed answers
→ Prepare comprehensive notes
```

---

## 🎯 Why This Architecture?

### Follows Specification EXACTLY
✅ Job-based temporary processing  
✅ No permanent storage  
✅ No login required  
✅ Auto-cleanup  
✅ 13-phase workflow  
✅ Classical NLP (no paid APIs)  
✅ Confidence score + Part A/B metrics

### Scalable & Maintainable
✅ Each phase is isolated  
✅ Pure functions where possible  
✅ ML logic separated from views  
✅ Testable components

### Cost-Effective
✅ No cloud storage costs  
✅ No API costs  
✅ Local models only  
✅ Automatic cleanup

---

## 📝 Development Notes

### Where Each Phase Lives

| Phase | File | Function/Class |
|-------|------|----------------|
| 1 | `core/guest_views.py` | `Phase1_Upload` |
| 2 | `analysis/services/pdf_extractor.py` | `Phase2_PDFDetection` |
| 3 | `analysis/services/extractor.py` | `Phase3_QuestionSegmentation` |
| 4 | `analysis/pipeline_13phases.py` | `Phase4_ModuleMapping` |
| 5 | `analysis/pipeline_13phases.py` | `Phase5_Normalization` |
| 6 | `analysis/services/embedder.py` | `Phase6_Embeddings` |
| 7 | `analysis/services/clustering.py` | `Phase7_Clustering` |
| 8-10 | `analysis/job_models.py` | `TempTopicCluster.calculate_metrics()` |
| 11 | `reports/ktu_report_generator_new.py` | `Phase11_PDFGeneration` |
| 12 | `templates/analysis/job_results.html` | Download buttons |
| 13 | `management/commands/cleanup_expired_jobs.py` | `Phase13_Cleanup` |

### Testing Individual Phases
```python
# Test Phase 2: PDF extraction
from apps.analysis.pipeline_13phases import Phase2_PDFDetection
result = Phase2_PDFDetection.detect_and_extract(paper)

# Test Phase 7: Clustering
from apps.analysis.pipeline_13phases import Phase7_Clustering
clusters = Phase7_Clustering.cluster_module(job, module_num=1)

# Test Phase 8: Priority scoring
cluster.calculate_metrics(total_years_in_job=7)
print(cluster.confidence_score)  # Should be 0-100
```

---

## 🔧 Configuration

### Media Storage (Temporary)
```python
# settings.py
MEDIA_ROOT = BASE_DIR / 'media'
MEDIA_URL = '/media/'

# Jobs auto-created at:
# media/jobs/<uuid>/pdfs/
# media/jobs/<uuid>/output/
```

### Job Expiry Settings
```python
# In AnalysisJob.mark_completed()
self.set_expiry(hours=24)  # Adjust as needed
```

### Clustering Parameters
```python
# In services/clustering.py
HDBSCANClustering(
    min_cluster_size=2,
    min_samples=1,
    metric='euclidean'
)
```

---

## 🎓 Credits

**Designed for:** KTU (Kerala Technological University) students  
**Architecture:** Follows classical NLP + rule-based approach  
**No AI Hype:** Pure engineering, no LLM fluff

---

## 📜 License

MIT License - Use freely, NO warranties.

---

**This is NOT a toy project. This is a production-ready system following software engineering best practices.**
