You are a senior backend engineer and NLP engineer.

Your task is to design and implement a Django-based web application called
“KTU Previous Year Question Priority Analyzer”.

This is NOT a toy project and NOT an LLM-based system.
Do NOT use OpenAI, GPT, or any paid API.
Use only classical NLP + sentence embeddings where explicitly required.

────────────────────────────────────
CORE GOAL
────────────────────────────────────
The system must allow WEB USERS (not admin) to upload multiple KTU previous-year
question paper PDFs and automatically generate MODULE-WISE PDFs that contain:

1) A COMPLETE question bank (all extracted questions, module-wise)
2) A PRIORITY ANALYSIS section that ranks REPEATED TOPICS based on:
   - Frequency across years
   - Marks weight
   - Combined Part A + Part B contribution
   - Confidence score (year coverage)

Priority classification is the CORE feature.
If priority is removed, the project becomes useless.

────────────────────────────────────
SCOPE & CONSTRAINTS
────────────────────────────────────
• KTU-specific exam pattern (fixed rules)
• Temporary job-based processing (NO permanent storage of user PDFs)
• Each upload session must be isolated using a job_id (UUID)
• Data must be auto-deleted after job completion or timeout
• Deterministic rules wherever possible
• AI ONLY for semantic similarity (topic clustering)

────────────────────────────────────
USER FLOW
────────────────────────────────────
1. User opens website
2. User uploads MULTIPLE PYQ PDFs (same subject)
3. System creates a job_id and temporary workspace
4. System analyzes PDFs
5. System generates 5 module-wise PDFs
6. User downloads PDFs
7. System auto-cleans all job data

NO login system is required.

────────────────────────────────────
TECH STACK (MANDATORY)
────────────────────────────────────
Backend:
• Django
• Python

PDF Processing:
• pdfplumber (text-based PDFs)
• PyMuPDF + pytesseract + OpenCV (scanned PDFs)

NLP / ML:
• sentence-transformers/all-MiniLM-L6-v2
• scikit-learn
• HDBSCAN
• NumPy

PDF Generation:
• ReportLab OR WeasyPrint

Frontend:
• Django templates
• Minimal JS
• Simple CSS

────────────────────────────────────
DETAILED WORKFLOW (MUST FOLLOW EXACTLY)
────────────────────────────────────

PHASE 1 — USER UPLOAD
• Accept multiple PDFs from user
• Validate file type and size
• Generate job_id (UUID)
• Create /media/jobs/<job_id>/

PHASE 2 — PDF TYPE DETECTION
• Detect whether PDF is text-based or scanned
• Extract raw text (store unchanged)
• Extract images separately

PHASE 3 — QUESTION SEGMENTATION (RULE-BASED)
• Detect PART A and PART B
• Extract logical questions:
  - Question number
  - Full question text
  - Marks
  - Year
  - Part (A or B)
• Handle OR questions and sub-parts correctly
• Each logical question = one semantic unit

PHASE 4 — MODULE MAPPING (RULE-BASED)
Use fixed KTU rules:
Qn 1–2   → Module 1
Qn 3–4   → Module 2
Qn 5–6   → Module 3
Qn 7–8   → Module 4
Qn 9–10  → Module 5
Qn 11–12 → Module 1
Qn 13–14 → Module 2
Qn 15–16 → Module 3
Qn 17–18 → Module 4
Qn 19–20 → Module 5

PHASE 5 — QUESTION NORMALIZATION
• Create a separate normalized text field
• Remove numbering, marks, year references
• Preserve academic meaning
• DO NOT overwrite raw text

PHASE 6 — EMBEDDING GENERATION
• Combine Part A + Part B questions
• Generate embeddings module-wise
• Cache embeddings per job

Model:
sentence-transformers/all-MiniLM-L6-v2

PHASE 7 — TOPIC CLUSTERING (CORE AI)
• Perform clustering PER MODULE
• Use HDBSCAN
• Each cluster = one exam topic
• Noise questions must be allowed

PHASE 8 — PRIORITY SCORING (CORE FEATURE)
For each topic cluster, compute:

• Frequency = number of DISTINCT YEARS appeared
• Average Marks
• Part A count
• Part B count

Priority Score Formula:
Priority Score = (2 × Frequency) + (Average Marks)

PHASE 9 — CONFIDENCE SCORE (MANDATORY EXTRA FEATURE)
Compute:
Confidence (%) =
(Number of years topic appeared ÷ Total years uploaded) × 100

PHASE 10 — PRIORITY TIERS
Assign tiers:
• Tier 1 – Very High Priority
• Tier 2 – High Priority
• Tier 3 – Medium Priority
• Tier 4 – Low Priority

PHASE 11 — MODULE-WISE PDF GENERATION
Generate ONE PDF PER MODULE.

Each PDF must contain:

SECTION A — COMPLETE QUESTION BANK
• PART A (year-wise grouping)
• PART B (year-wise grouping)
• Preserve question text, marks, diagrams

SECTION B — REPEATED QUESTION ANALYSIS
• Tier-wise topics
• Repetition count
• Appearing years
• Confidence score
• Part A vs Part B contribution

FINAL STUDY PRIORITY ORDER
• Linear list from Tier 1 → Tier 4

The PDF structure must match standard KTU exam formatting.

PHASE 12 — USER DELIVERY
• Show download buttons for Module 1–5 PDFs

PHASE 13 — AUTO CLEANUP
• Delete all job data after download or timeout

────────────────────────────────────
ARCHITECTURE REQUIREMENTS
────────────────────────────────────
• Separate modules for:
  - PDF extraction
  - Question parsing
  - Module mapping
  - Embeddings
  - Clustering
  - Priority scoring
  - PDF generation
• DO NOT put ML logic inside Django views
• Views must only orchestrate workflow



────────────────────────────────────
OUTPUT EXPECTATION
────────────────────────────────────
Provide:
1. Django project structure
2. Models / data structures (job-based)
3. Views + URL flow
4. Core pipeline pseudo-code
5. Notes on where each phase is implemented



# 🥇 SELECTED EXTRA FEATURE (LOCKED)

## ✅ **Confidence Score + Part-A / Part-B Contribution (COMBINED FEATURE)**

This is the **highest ROI feature** you can add.
It upgrades your priority system from *ranking* → *quantified intelligence*.

No ML risk. No hype. Pure logic.

---

# 🧠 WHAT THIS FEATURE ACTUALLY ADDS

For **each PRIORITY TOPIC**, you will now show:

1. **Priority Tier** (already there)
2. **Confidence %** (NEW)
3. **Part A vs Part B split** (NEW)

This answers the student’s real questions:

* *How sure is this topic?*
* *Is it usually short notes or long answers?*

Most projects NEVER answer this.

---

# 📐 EXACT DEFINITIONS (NO AMBIGUITY)

## 1️⃣ Confidence Score

### Formula (simple, defendable):

```
Confidence (%) =
(Number of distinct years topic appeared ÷
 Total years uploaded) × 100
```

### Example:

* Topic appeared in **6 out of 7 years**
* Confidence = **85.7%**

This is **not guessing**.
This is probability from historical data.

---

## 2️⃣ Part A vs Part B Contribution

For each topic cluster:

```
Part A appearances = count of questions from Part A
Part B appearances = count of questions from Part B
```

Then display:

```
Appears as:
• Part A: 4 times
• Part B: 3 times
```

This is **exam intelligence**, not AI fluff.

---

# 🧩 WHERE THIS FITS IN YOUR WORKFLOW

We are NOT adding a new phase.
We are **enhancing Phase 9 (Priority Scoring)**.

---

## 🔁 UPDATED PHASE 9 — PRIORITY SCORING (FINAL)

For each **topic cluster**:

### Step 9.1 — Frequency

* Count distinct years (unchanged)

### Step 9.2 — Marks influence

* Average marks (unchanged)

### 🔥 Step 9.3 — NEW: Part-wise contribution

* Count Part A questions
* Count Part B questions

### 🔥 Step 9.4 — NEW: Confidence

* Use total years uploaded in job

### Step 9.5 — Priority score

```
Priority Score = (2 × Frequency) + (Avg Marks)
```

Tier assignment stays same.

Nothing breaks. Everything improves.

---

# 🖨️ HOW IT APPEARS IN THE OUTPUT PDF

### Example (TOP PRIORITY topic):

```
1. Disaster Risk Management (Framework + Core Elements)

Appears in: 2021, 2022, 2023, 2024, 2025
Repetition count: 6
Confidence: 85%

Appears as:
• Part A: 2 times
• Part B: 4 times

→ Very high probability long-answer topic.
```
