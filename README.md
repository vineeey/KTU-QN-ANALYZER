# PYQ Analyzer

An intelligent Django web application that analyzes Previous Year Question Papers (PYQs), identifies repeated topics, assigns priority levels, and generates comprehensive module-wise PDF reports to help students prepare efficiently for exams.

## 🎯 Key Features

- **Batch PDF Upload**: Upload multiple PYQ PDFs at once
- **Automatic Question Extraction**: Extracts questions from Part A and Part B
- **Smart Module Classification**: Maps questions to modules using configurable patterns
- **Topic Clustering**: Groups similar questions across years
- **Priority Assignment**: 4-tier system based on repetition frequency
  - Tier 1 (Top Priority): 4+ appearances
  - Tier 2 (High Priority): 3 appearances
  - Tier 3 (Medium Priority): 2 appearances
  - Tier 4 (Low Priority): 1 appearance
- **Module-Wise PDF Reports**: Downloadable PDFs with:
  - Questions grouped by year
  - Repeated question analysis
  - Study priority order
  - Strategy recommendations
- **Interactive Analytics Dashboard**: Charts and graphs with Chart.js
- **Zero-Cost Solution**: SQLite database, no external APIs

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 8GB RAM (minimum)
- No GPU required

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vineeey/pyq-analyzer.git
cd pyq-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run migrations**
```bash
python manage.py migrate
```

4. **Create test data (optional)**
```bash
python manage.py setup_test_data
```

This creates:
- Admin user: `admin@test.com` / `admin123`
- Sample subject: Disaster Management (MCN301)
- 5 modules with topics and keywords

5. **Start Django-Q worker** (for background tasks)
```bash
# In a separate terminal
python manage.py qcluster
```

6. **Run development server**
```bash
python manage.py runserver
```

7. **Access the application**
```
http://localhost:8000
```

## 📖 Usage Guide

### 1. Create a Subject
- Navigate to **Subjects** → **Create New**
- Enter subject name, code, university details
- System auto-creates 5 modules (customizable)

### 2. Configure Exam Pattern (Optional)
- Default KTU pattern is pre-configured
- Part A: Q1-Q10 mapped to Modules 1-5
- Part B: Q11-Q20 mapped to Modules 1-5

### 3. Upload Question Papers
- Go to **Papers** → **Upload**
- Select multiple PDF files
- Files are processed automatically in background
- Supported format: Text-based PDFs (scanned PDFs may have issues)

### 4. Analyze Topics
- Once papers are processed, visit **Analytics**
- Click **"Analyze Topics"** button
- Wait for clustering to complete (1-2 minutes typically)

### 5. View Analytics
- **Subject Dashboard**: Overview and top topics across modules
- **Module Analytics**: Detailed frequency charts per module
- Color-coded priority visualization

### 6. Download Reports
- Go to **Reports** section
- Download individual module PDFs
- Each PDF contains:
  - Part A questions by year
  - Part B questions by year
  - Repeated question analysis with priorities
  - Final study priority order

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│   PDF PYQs  │
└──────┬──────┘
       │
       v
┌─────────────────────┐
│  Extract Questions  │
│  (pdfplumber/PyMuPDF)│
└──────┬──────────────┘
       │
       v
┌────────────────────┐
│ Classify to Modules│
│ (Pattern/Keywords) │
└──────┬─────────────┘
       │
       v
┌───────────────────┐
│ Cluster Topics    │
│ (Similarity Match)│
└──────┬────────────┘
       │
       v
┌──────────────────┐
│ Assign Priorities│
│ (Frequency Count)│
└──────┬───────────┘
       │
       v
┌───────────────────┐
│Generate PDF Reports│
│   (WeasyPrint)    │
└───────────────────┘
```

## 🗂️ Project Structure

```
pyq-analyzer/
├── apps/
│   ├── subjects/      # Subject & Module management
│   ├── papers/        # PDF uploads
│   ├── questions/     # Question storage
│   ├── analysis/      # Extraction & classification
│   ├── analytics/     # Topic clustering
│   └── reports/       # PDF generation
├── templates/         # HTML templates
├── static/            # CSS, JS, images
├── media/             # Uploaded files, generated PDFs
└── manage.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db/pyq_analyzer.sqlite3
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Exam Pattern Configuration
Edit in Django admin or via ExamPattern model:
```python
pattern_config = {
    'part_a': {'1': 1, '2': 1, '3': 2, ...},
    'part_b': {'11': 1, '12': 1, '13': 2, ...}
}
```

## 🎨 Technology Stack

- **Backend**: Django 5.0+
- **Database**: SQLite3
- **Task Queue**: Django-Q2
- **PDF Processing**: pdfplumber, PyMuPDF
- **PDF Generation**: WeasyPrint
- **ML**: sentence-transformers (optional)
- **Frontend**: Tailwind CSS, Chart.js
- **Icons**: Lucide

## 📊 Performance

- **Hardware**: Tested on HP 15s (Ryzen 3 3500U, 8GB RAM)
- **Upload**: 5-10 PDFs in ~30 seconds
- **Extraction**: ~5-10 seconds per paper
- **Clustering**: ~1-2 minutes for 50-100 questions
- **PDF Generation**: ~2-3 seconds per module

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

[Add your license]

## 🙏 Acknowledgments

Designed for KTU (APJ Abdul Kalam Technological University) question papers,
but adaptable to any university or exam board.

## 📧 Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/vineeey/pyq-analyzer/issues)
- Email: [Add contact email]

---

**Built with ❤️ for students preparing for exams**
