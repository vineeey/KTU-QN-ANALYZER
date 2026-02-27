"""
KTU Module Report Generator - EXACT MATCH to Expected Output
Generates PDFs with proper formatting, colors, emojis, and structure.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict
from django.conf import settings
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

from apps.subjects.models import Subject, Module
from apps.questions.models import Question
from apps.analytics.models import TopicCluster

logger = logging.getLogger(__name__)


class KTUModuleReportGenerator:
    """Generates KTU-style module reports matching expected output format."""
    
    def __init__(self, subject: Subject, output_dir: Optional[str] = None):
        self.subject = subject
        self.output_dir = output_dir  # Optional override for output directory
        self.diamond = "◆"  # Red diamond bullet for years
    
    def generate_all_module_reports(self) -> Dict[int, Optional[str]]:
        """Generate reports for all 5 modules."""
        results = {}
        for module_num in range(1, 6):
            module = Module.objects.filter(subject=self.subject, number=module_num).first()
            if module:
                pdf_path = self.generate_module_report(module)
                results[module_num] = pdf_path
        return results
    
    def generate_module_report(self, module: Module) -> Optional[str]:
        """Generate single module PDF report."""
        try:
            # Output path — use explicit override if given, else default media location
            if self.output_dir:
                output_dir = Path(self.output_dir)
            else:
                output_dir = Path(settings.MEDIA_ROOT) / 'reports' / str(self.subject.id)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"Module {module.number}.pdf"
            
            # Create PDF
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            data = self._prepare_data(module)
            story = self._build_content(data)
            
            # Generate
            doc.build(story)
            logger.info(f"Generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            return None
    
    def _prepare_data(self, module: Module) -> Dict[str, Any]:
        """Prepare all data for the report."""
        # Get all questions for this module
        questions = Question.objects.filter(
            module=module,
            paper__subject=self.subject
        ).select_related('paper').order_by('paper__year', 'question_number')
        
        # Calculate question numbers
        part_a_start = (module.number - 1) * 2 + 1
        part_a_end = part_a_start + 1
        part_b_start = 10 + (module.number - 1) * 2 + 1
        part_b_end = part_b_start + 1
        
        # Group questions
        part_a = self._group_part_a(questions.filter(part='A'))
        part_b = self._group_part_b(questions.filter(part='B'))
        
        # Analyze priorities based on question frequency (without TopicCluster)
        priority = self._analyze_priorities_from_questions(questions)
        
        return {
            'module': module,
            'subject': self.subject,
            'part_a_start': part_a_start,
            'part_a_end': part_a_end,
            'part_b_start': part_b_start,
            'part_b_end': part_b_end,
            'part_a': part_a,
            'part_b': part_b,
            'priority': priority,
        }
    
    def _build_content(self, data: Dict) -> List:
        """Build PDF content."""
        styles = getSampleStyleSheet()
        story = []
        
        # === TITLE ===
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=black
        )
        story.append(Paragraph(
            f"Module {data['module'].number} – DISASTER MANAGEMENT (KTU 2019 Scheme)",
            title_style
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # === PART A ===
        heading_style = ParagraphStyle(
            'Heading',
            fontSize=14,
            fontName='Helvetica-Bold',
            spaceAfter=8,
            spaceBefore=10,
            textColor=black
        )
        story.append(Paragraph("PART A (3 Marks each)", heading_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            fontSize=9,
            textColor=HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=15
        )
        story.append(Paragraph(
            f"(Qn {data['part_a_start']} & {data['part_a_end']} from all papers belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        # Part A Questions
        year_style = ParagraphStyle(
            'YearHeading',
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12,
            textColor=HexColor('#CC0000')  # Red
        )
        
        question_style = ParagraphStyle(
            'Question',
            fontSize=10,
            leftIndent=15,
            spaceAfter=4
        )
        
        for year_data in data['part_a']:
            # Red diamond + year
            story.append(Paragraph(
                f'<font color="#CC0000">◆</font> <b>{year_data["year"]}</b>',
                year_style
            ))
            
            for q in year_data['questions']:
                story.append(Paragraph(
                    f"• {q['text']} — ({q['year_short']}, {q['marks']} marks)",
                    question_style
                ))
        
        story.append(Spacer(1, 0.5*cm))
        
        # === PART B ===
        story.append(Paragraph("PART B (14 Marks each)", heading_style))
        story.append(Paragraph(
            f"(Qn {data['part_b_start']} & {data['part_b_end']} belong to Module {data['module'].number})",
            subtitle_style
        ))
        
        qn_number_style = ParagraphStyle(
            'QnNumber',
            fontSize=10,
            fontName='Helvetica-Bold',
            spaceBefore=10,
            spaceAfter=4
        )
        
        for year_data in data['part_b']:
            # Red diamond + year
            story.append(Paragraph(
                f'<font color="#CC0000">◆</font> <b>{year_data["year"]}</b>',
                year_style
            ))
            
            for qn in year_data['questions']:
                story.append(Paragraph(f"Qn {qn['number']}", qn_number_style))
                
                for part in qn['parts']:
                    # Split OR parts into separate bullets
                    import re
                    or_parts = re.split(r'\s+OR\s+', part['text'], flags=re.IGNORECASE)
                    
                    for or_part in or_parts:
                        or_part = or_part.strip()
                        if or_part:
                            story.append(Paragraph(
                                f"• {or_part} — ({part['year_short']}, {part['marks']} marks)",
                                question_style
                            ))
        
        # === PAGE BREAK ===
        story.append(PageBreak())
        
        # === REPEATED QUESTIONS ANALYSIS ===
        section_title_style = ParagraphStyle(
            'SectionTitle',
            fontSize=14,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=10,
            spaceBefore=15,
            textColor=black
        )
        
        story.append(Paragraph(
            f"■ Module {data['module'].number} – Repeated Questions<br/>(Prioritized List)",
            section_title_style
        ))
        story.append(Paragraph(
            "<i>(Highest repeated = highest priority)</i>",
            subtitle_style
        ))
        story.append(Spacer(1, 0.5*cm))
        
        # Priority tiers
        tier_title_style = ParagraphStyle(
            'TierTitle',
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceBefore=15,
            spaceAfter=8,
            textColor=black
        )

        topic_style = ParagraphStyle(
            'Topic',
            fontSize=10,
            leftIndent=15,
            spaceAfter=3
        )

        years_style = ParagraphStyle(
            'Years',
            fontSize=9,
            leftIndent=15,
            spaceAfter=6,
            textColor=HexColor('#555555')
        )

        hint_style = ParagraphStyle(
            'Hint',
            fontSize=9,
            leftIndent=25,
            spaceAfter=2,
            textColor=HexColor('#444444')
        )

        def _render_tier(label: str, items: list):
            if not items:
                return
            story.append(Paragraph(label, tier_title_style))
            for item in items:
                story.append(Paragraph(
                    f"<b>{item['rank']}. {item['topic']}</b>",
                    topic_style
                ))
                story.append(Paragraph(
                    f"Appears in: {item['years']}",
                    years_style
                ))
                for hint in item.get('bullet_hints', []):
                    if hint and hint.strip():
                        story.append(Paragraph(f'• "{hint.strip()}"', hint_style))

        # TOP PRIORITY
        _render_tier('🔥 <b>TOP PRIORITY — Repeated 4+ Times</b>', data['priority']['top'])

        # HIGH PRIORITY
        _render_tier('🔥 <b>HIGH PRIORITY — Repeated 3 Times</b>', data['priority']['high'])

        # MEDIUM PRIORITY
        _render_tier('⚡ <b>MEDIUM PRIORITY — Repeated 2 Times</b>', data['priority']['medium'])

        # LOW PRIORITY
        _render_tier(
            f'✓ <b>LOW PRIORITY — Appears Only Once (But still Module {data["module"].number})</b>',
            data['priority']['low']
        )

        # === FINAL PRIORITIZED STUDY ORDER ===
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            '📌 <b>FINAL PRIORITIZED STUDY ORDER</b><br/>(My recommended ranking)',
            section_title_style
        ))
        story.append(Paragraph(
            "<i>If you want to score high, study in THIS order:</i>",
            subtitle_style
        ))
        story.append(Spacer(1, 0.3*cm))
        
        tier_heading_style = ParagraphStyle(
            'TierHeading',
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6
        )
        
        list_style = ParagraphStyle(
            'ListItem',
            fontSize=10,
            leftIndent=20,
            spaceAfter=3
        )
        
        # Study order tiers
        if data['priority']['top']:
            story.append(Paragraph(
                "<b>Tier 1 (Most repeated — must learn first)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['top'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['high']:
            story.append(Paragraph(
                "<b>Tier 2 (Frequently repeated)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['high'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['medium']:
            story.append(Paragraph(
                "<b>Tier 3 (Moderately repeated)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['medium'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        if data['priority']['low']:
            story.append(Paragraph(
                "<b>Tier 4 (One-time but possible)</b>",
                tier_heading_style
            ))
            for i, item in enumerate(data['priority']['low'], 1):
                story.append(Paragraph(f"{i}. {item['topic']}", list_style))
        
        return story
    
    def _group_part_a(self, questions) -> List[Dict]:
        """Group Part A questions by year."""
        by_year = defaultdict(list)
        
        for q in questions:
            year = self._format_year(q.paper)
            # Prefer LLM-cleaned text; fall back to raw OCR text
            text = self._clean_text(q.cleaned_text or q.text)
            # Skip questions that are clearly page headers / OCR junk
            if not text or len(text) < 8 or self._is_junk(text):
                continue
            by_year[year].append({
                'text': text,
                'marks': q.marks or 3,
                'year_short': self._short_year(q.paper)
            })
        
        result = []
        for year in self._sorted_years(by_year.keys()):
            result.append({
                'year': year,
                'questions': by_year[year]
            })
        return result
    
    def _group_part_b(self, questions) -> List[Dict]:
        """Group Part B questions by year and question number."""
        by_year = defaultdict(lambda: defaultdict(list))
        
        for q in questions:
            year = self._format_year(q.paper)
            qn = q.question_number
            text = self._clean_text(q.cleaned_text or q.text)
            if not text or len(text) < 8 or self._is_junk(text):
                continue
            by_year[year][qn].append({
                'text': text,
                'marks': q.marks or 7,
                'year_short': self._short_year(q.paper)
            })
        
        result = []
        for year in self._sorted_years(by_year.keys()):
            questions_list = []
            for qn in sorted(by_year[year].keys()):
                questions_list.append({
                    'number': qn,
                    'parts': by_year[year][qn]
                })
            result.append({
                'year': year,
                'questions': questions_list
            })
        return result
    
    def _analyze_priorities(self, clusters) -> Dict[str, List]:
        """Analyze topic clusters and create priority tiers."""
        priority = {
            'top': [],     # 4+ times
            'high': [],    # 3 times
            'medium': [],  # 2 times
            'low': []      # 1 time
        }
        
        rank = 1
        for cluster in clusters:
            freq = cluster.frequency_count
            years = cluster.years_appeared or []
            
            item = {
                'rank': rank,
                'topic': cluster.topic_name,
                'frequency': freq,
                'years': ', '.join(str(y) for y in years) if years else 'N/A'
            }
            
            if freq >= 4:
                priority['top'].append(item)
            elif freq == 3:
                priority['high'].append(item)
            elif freq == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
            
            rank += 1
        
        return priority
    
    def _analyze_priorities_from_questions(self, questions) -> Dict[str, List]:
        """
        Group semantically similar questions across years and assign priority tiers.

        Strategy
        --------
        1. Encode every question with sentence-transformers (singleton model).
        2. Build a cosine-similarity matrix.
        3. Greedily merge questions whose similarity > THRESHOLD into topic groups.
        4. Name each group after its longest / most descriptive question.
        5. Count distinct exam sessions (years_appeared or paper.year).
        6. Bucket into TOP / HIGH / MEDIUM / LOW by cross-year frequency.
        """
        from collections import defaultdict
        import numpy as np

        SIMILARITY_THRESHOLD = 0.72   # tune: lower = more aggressive merging

        priority: Dict[str, List] = {
            'top': [], 'high': [], 'medium': [], 'low': []
        }

        qs_list = list(questions)
        if not qs_list:
            return priority

        # Use cleaned_text when available; fall back to raw OCR text
        texts = [(q.cleaned_text or q.text).strip() for q in qs_list]
        # Filter out junk entries before encoding
        valid_mask = [not self._is_junk(t) and len(t) >= 8 for t in texts]
        qs_list = [q for q, ok in zip(qs_list, valid_mask) if ok]
        texts    = [t for t, ok in zip(texts, valid_mask) if ok]
        if not qs_list:
            return priority

        # ── Encode ────────────────────────────────────────────────────────
        embeddings = self._encode_texts(texts)

        if embeddings is None:
            # Fallback: group by exact first-80-char key
            return self._priority_exact_fallback(qs_list)

        # ── Cosine similarity matrix ───────────────────────────────────────
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T   # (N, N)

        # ── Greedy union-find clustering ───────────────────────────────────
        n = len(qs_list)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                    union(i, j)

        # ── Collect groups ─────────────────────────────────────────────────
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        # ── Compute year frequency per group ───────────────────────────────
        def _get_year(q) -> Optional[str]:
            if q.years_appeared:
                return q.years_appeared[0] if isinstance(q.years_appeared, list) else str(q.years_appeared)
            if q.paper and q.paper.year:
                return str(q.paper.year)
            if q.paper and q.paper.exam_type:
                return q.paper.exam_type
            return None

        group_data = []
        for root, members in groups.items():
            member_qs = [qs_list[i] for i in members]
            years = sorted({y for q in member_qs for y in ([_get_year(q)] if _get_year(q) else [])})
            # Representative name: use cleaned_text, pick the longest
            rep_text = max(
                ((q.cleaned_text or q.text).strip() for q in member_qs),
                key=len
            )
            group_data.append({
                'members': member_qs,
                'years': years,
                'year_count': len(years),
                'topic': rep_text[:120] + ('…' if len(rep_text) > 120 else ''),
            })

        # Sort by frequency (descending), then by first year (ascending)
        group_data.sort(key=lambda g: (-g['year_count'], g['years'][0] if g['years'] else ''))

        # ── Assign tiers ───────────────────────────────────────────────────
        rank = 1
        for gd in group_data:
            yrs = gd['years']
            fc  = gd['year_count']
            item = {
                'rank': rank,
                'topic': gd['topic'],
                'frequency': fc,
                'years': ', '.join(yrs) if yrs else 'N/A',
                'bullet_hints': [(q.cleaned_text or q.text)[:80] for q in gd['members'][:4]],
            }
            if fc >= 4:
                priority['top'].append(item)
            elif fc == 3:
                priority['high'].append(item)
            elif fc == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
            rank += 1
            if rank > 25:
                break

        return priority

    # ── Sentence-transformer singleton ────────────────────────────────────
    _st_model = None

    def _encode_texts(self, texts: List[str]):
        """
        Encode texts with sentence-transformers.
        Returns numpy array of shape (N, D) or None on failure.
        """
        try:
            import numpy as np
            if KTUModuleReportGenerator._st_model is None:
                from sentence_transformers import SentenceTransformer
                KTUModuleReportGenerator._st_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            vecs = KTUModuleReportGenerator._st_model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            logger.warning("Semantic encoding failed (%s) – using exact-match fallback", exc)
            return None

    def _priority_exact_fallback(self, questions) -> Dict[str, List]:
        """Exact-text deduplication fallback (no ML required)."""
        from collections import Counter, defaultdict

        priority: Dict[str, List] = {'top': [], 'high': [], 'medium': [], 'low': []}
        counter: Counter = Counter()
        years_map: dict = defaultdict(set)

        for q in questions:
            key = (q.cleaned_text or q.text).strip()[:100]
            if self._is_junk(key) or len(key) < 8:
                continue
            counter[key] += 1
            yr = q.paper.year if q.paper and q.paper.year else None
            if yr:
                years_map[key].add(str(yr))

        rank = 1
        for text, freq in counter.most_common(20):
            years = sorted(years_map[text])
            item = {
                'rank': rank,
                'topic': text[:80] + ('…' if len(text) > 80 else ''),
                'frequency': freq,
                'years': ', '.join(years) if years else 'N/A',
            }
            fc = len(years) if years else freq
            if fc >= 4:
                priority['top'].append(item)
            elif fc == 3:
                priority['high'].append(item)
            elif fc == 2:
                priority['medium'].append(item)
            else:
                priority['low'].append(item)
            rank += 1
        return priority
    
    def _format_year(self, paper) -> str:
        """Format year as 'December 2021'."""
        if paper and paper.exam_type:
            return paper.exam_type
        return 'Unknown'
    
    def _short_year(self, paper) -> str:
        """Format year as 'Dec 2021'."""
        if not paper or not paper.exam_type:
            return ''
        
        months = {
            'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
            'April': 'Apr', 'May': 'May', 'June': 'Jun',
            'July': 'Jul', 'August': 'Aug', 'September': 'Sep',
            'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
        }
        
        year_str = paper.exam_type
        for full, short in months.items():
            if full in year_str:
                return year_str.replace(full, short)
        return year_str
    
    def _clean_text(self, text: str) -> str:
        """Clean question text."""
        import re
        text = text.strip()
        # Remove leading numbers/formatting
        text = re.sub(r'^[Il1O0-9\.\)\s]+', '', text).strip()
        return text

    def _is_junk(self, text: str) -> bool:
        """Return True if the text is an OCR artifact / page header / unrecoverable garbage."""
        import re
        if not text or len(text.strip()) < 8:
            return True
        junk_patterns = [
            r'^\s*[A-Z]\s+subcontinent',          # "Z subcontinent. -,"
            r'PART\s*[B8]\s*[\{\[]',               # "PART B {Answer one full..."
            r'ea€\*\s*module',                     # OCR page header fragment
            r'^[\W\s]{0,3}$',                      # Only symbols/spaces
            r'ltaz[-\s]*afi',                      # "ltaz-afi,mapping"
            r'!a\}[€}]r',                          # "!a}€r"
        ]
        for pat in junk_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False
    
    def _sorted_years(self, years) -> List[str]:
        """Sort years chronologically."""
        year_order = {
            'December 2021': 1,
            'December 2022': 2,
            'December 2023': 3,
            'June 2024': 4,
            'November 2024': 5,
            'May 2025': 6
        }
        return sorted(years, key=lambda y: year_order.get(y, 999))


def generate_ktu_module_reports(subject: Subject) -> Dict[int, Optional[str]]:
    """Generate all module reports."""
    generator = KTUModuleReportGenerator(subject)
    return generator.generate_all_module_reports()
