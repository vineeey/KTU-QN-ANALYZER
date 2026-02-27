# KTU QN Analyzer — Production System

An intelligent, production-grade Django web application that processes **scanned image-based KTU question papers**, extracts structured questions, performs semantic clustering, assigns study priorities, and generates module-wise PDF reports.

> **CPU-only · No GPU required · 8 GB RAM minimum**

---

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| Scanned PDF OCR | PaddleOCR with per-page confidence retry |
| Image Preprocessing | OpenCV: grayscale, median blur, sharpening, adaptive threshold, deskew |
| Text Cleaning | Ligature normalisation, artefact removal, optional LLM hook |
| Question Segmentation | Stateful Part A/B parser – number, marks, sub-questions, module hints |
| Module Classification | Stage 1: keyword/topics · Stage 2: sentence-transformer cosine similarity |
| Semantic Clustering | AgglomerativeClustering (default) · HDBSCAN (optional) |
| Priority Assignment | 4-tier based on cross-year repetition frequency (configurable) |
| PDF Reports | WeasyPrint + Tailwind CSS templates, per-module and full-subject |
| Hybrid LLM | Gemini → Qwen → Ollama fallback chain for OCR cleaning |
| Background Processing | Django-Q2 async task queue (10-step pipeline) |

---

## 🤖 Hybrid LLM Pipeline

The system uses a **graceful-fallback LLM chain** — classification and extraction work even with no LLM configured:

```
Request
  │
  ├─► 1st: Google Gemini (gemini-2.0-flash-lite)   ← set GEMINI_API_KEY
  ├─► 2nd: Qwen via DashScope (qwen2.5-7b-instruct) ← set QWEN_API_KEY
  ├─► 3rd: Ollama local    (qwen2.5:7b-instruct)    ← run ollama locally
  └─► Fallback: rule-based extraction only
```

---

## 🔄 10-Step Processing Pipeline

Each uploaded PDF is processed asynchronously via this pipeline:

```
Step 1  ─ PDF → page images (300 DPI, PyMuPDF)
Step 2  ─ Image preprocessing (OpenCV)
Step 3  ─ OCR (PaddleOCR, with low-confidence retry)
Step 4  ─ Text cleaning (artefact removal + LLM hook)
Step 5  ─ Question segmentation (stateful Part A/B parser)
Step 6  ─ Module classification (keyword → semantic → LLM)
Step 7  ─ Embedding generation  (sentence-transformers, cached)
         — subject-level steps (on demand) —
Step 8  ─ Semantic clustering   (AgglomerativeClustering / HDBSCAN)
Step 9  ─ Priority assignment   (4-tier frequency-based)
Step 10 ─ Analytics counters update
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8 GB RAM (minimum)
- No GPU required
- (Optional) Ollama for fully offline LLM support

### Installation

```bash
# 1. Clone
git clone https://github.com/vineeey/ktu-qn-analyzer.git
cd ktu-qn-analyzer

# 2. Virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment
cp .env.example .env
# Edit .env — add GEMINI_API_KEY for best results

# 5. Database
python manage.py migrate

# 6. (Optional) Seed test data
python manage.py setup_test_data
# Creates: admin@test.com / admin123, sample subject MCN301

# 7. Start background worker (required — separate terminal)
python manage.py qcluster

# 8. Run server
python manage.py runserver
```

Access at **http://localhost:8000**

---

## 📖 Usage Guide

### 1. Create a Subject
**Subjects → Create New** — enter name, code, and select KTU.

### 2. Upload Question Papers
**Papers → Upload** — select multiple scanned PDFs.
Processing begins automatically in the background.

### 3. Monitor Progress
Papers page shows: Rendering → OCR → Segmenting → Classifying → Done.

### 4. Run Topic Analysis
**Analytics → Analyze Topics** — triggers clustering + priority assignment.

### 5. View Analytics Dashboard
- Module distribution chart
- Priority tier breakdown
- Cluster frequency charts
- REST endpoints for incremental reanalysis

### 6. Download Reports
**Reports** → per-module PDF with:
- Part A questions by year
- Part B questions by year
- Repeated question analysis + priority tiers
- Study order recommendations

---

## 🗂️ Project Structure

```
ktu-qn-analyzer/
├── apps/
│   ├── subjects/          # Subject & Module management
│   ├── papers/            # PDF upload, PaperPage model
│   ├── questions/         # Question, QuestionEmbeddingCache
│   ├── analysis/          # 10-step extraction pipeline
│   │   ├── services/
│   │   │   ├── ocr_engine.py          # PaddleOCR wrapper
│   │   │   ├── image_preprocessor.py  # OpenCV functions
│   │   │   ├── text_cleaner.py        # OCR text cleaning
│   │   │   ├── segmenter.py           # Stateful question parser
│   │   │   ├── classifier.py          # Two-stage module classifier
│   │   │   └── hybrid_llm_service.py  # Gemini/Qwen/Ollama chain
│   │   ├── tasks.py                   # Django-Q2 pipeline tasks
│   │   └── pipeline.py                # Orchestration (legacy compat.)
│   ├── analytics/         # Clustering, priority, dashboards
│   │   ├── models.py      # ClusterGroup, ClusterMembership, PriorityAssignment
│   │   └── services/
│   │       ├── clustering.py          # AgglomerativeClustering / HDBSCAN
│   │       └── priority_engine.py     # 4-tier priority assignment
│   ├── reports/           # PDF report generation
│   │   ├── models.py      # GeneratedReport
│   │   └── services/
│   │       └── report_generator.py    # WeasyPrint HTML→PDF
│   ├── rules/             # KTU exam pattern rules engine
│   ├── users/             # Authentication & user management
│   └── core/              # Shared base models, mixins, utilities
├── services/
│   ├── llm/               # Low-level LLM provider clients
│   └── embedding/         # Sentence-transformer helpers
├── templates/             # Tailwind CSS HTML templates
├── static/                # Chart.js, Lucide icons, CSS
├── media/                 # Uploaded PDFs, generated reports
├── docs/                  # LLM config & hybrid usage docs
├── scripts/               # Ollama setup, model downloads
├── config/                # Django settings & URL config
└── manage.py
```

---

## 🗃️ Data Models

| Model | App | Purpose |
|-------|-----|---------|
| `Subject` | subjects | University subject with modules |
| `Module` | subjects | Chapter/unit within a subject |
| `Paper` | papers | Uploaded question paper (PDF) |
| `PaperPage` | papers | Individual page image + OCR text |
| `Question` | questions | Extracted question with metadata |
| `QuestionEmbeddingCache` | questions | Cached sentence-transformer vector |
| `ClusterGroup` | analytics | Group of semantically similar questions |
| `ClusterMembership` | analytics | Question → ClusterGroup join |
| `PriorityAssignment` | analytics | Tier (1–4) for a ClusterGroup |
| `GeneratedReport` | reports | PDF report file record |
| `TopicCluster` | analytics | Legacy topic cluster (retained) |

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
# Django
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db/pyq_analyzer.sqlite3

# LLM — Gemini (primary, recommended)
GEMINI_API_KEY=your-gemini-api-key

# LLM — Qwen via Alibaba DashScope (secondary)
QWEN_API_KEY=your-qwen-or-hf-token
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen2.5-7b-instruct

# LLM — Ollama (local, offline fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_USE_HYBRID=true
SIMILARITY_EMBEDDING_MODEL=multi-qa-MiniLM-L6-cos-v1
SIMILARITY_THRESHOLD_HIGH=0.85
SIMILARITY_THRESHOLD_LOW=0.65

# OCR enhancement
OCR_USE_LLM_CLEANING=true
OCR_BATCH_PAGES=true

# Priority tier thresholds (optional — defaults shown)
PRIORITY_TIER_1_THRESHOLD=4
PRIORITY_TIER_2_THRESHOLD=3
PRIORITY_TIER_3_THRESHOLD=2
```

### Ollama Setup (local, fully offline)

```bash
bash scripts/setup_ollama.sh
# or:
ollama pull qwen2.5:7b-instruct
```

---

## 🎨 Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Django 5.0+, Python 3.12+ |
| Database | SQLite3 (production: PostgreSQL) |
| Task Queue | Django-Q2 |
| OCR | PaddleOCR + PaddlePaddle (CPU) |
| Image Processing | OpenCV, Pillow |
| PDF Parsing | PyMuPDF, pdfplumber |
| PDF Generation | WeasyPrint |
| ML / Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Clustering | scikit-learn AgglomerativeClustering, HDBSCAN |
| LLM (optional) | Google Gemini, Qwen, Ollama |
| Frontend | Tailwind CSS, Chart.js, Lucide Icons |

---

## 📊 Performance

| Operation | Time |
|-----------|------|
| PDF upload (5–10 papers) | ~30 s |
| OCR per page | ~0.5–1.5 s |
| Extraction per paper | ~10–20 s |
| AI clustering (100 questions) | ~1–2 min |
| PDF report generation | ~2–3 s |

*Tested on HP 15s, Ryzen 3 3500U, 8 GB RAM (CPU only)*

---

## 🧪 Running Tests

```bash
# All tests
pytest

# Specific service
pytest apps/analysis/tests/test_segmenter.py -v
pytest apps/analysis/tests/test_text_cleaner.py -v
pytest apps/analysis/tests/test_classifier.py -v
pytest apps/analytics/tests/test_priority_engine.py -v

# With coverage
pytest --cov=apps --cov-report=term-missing
```

---

## 🏗️ Engineering Standards

- **Clean architecture**: views ← services ← models (no ML in views)
- **Singleton models**: embedding model loaded once per process
- **Batch encoding**: questions encoded in configurable batches (64)
- **Incremental processing**: cached embeddings never recomputed
- **Centralised logging**: every service uses `logging.getLogger(__name__)`
- **Graceful fallback**: each pipeline step catches and logs exceptions
- **Environment config**: all secrets and thresholds via `.env`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests before submitting: `pytest`
4. Submit a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built for KTU (APJ Abdul Kalam Technological University) exam preparation.
Adaptable to any university with configurable exam patterns and module keywords.

---

**Built with ❤️ for KTU students preparing for exams**
