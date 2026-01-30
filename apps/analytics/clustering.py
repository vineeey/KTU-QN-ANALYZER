"""
AI-Powered Topic Clustering and Repetition Analysis Service.
Uses sentence transformers for semantic similarity to group questions by meaning.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from django.db import transaction

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback clustering")

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using greedy clustering")

from apps.questions.models import Question
from apps.subjects.models import Subject, Module
from apps.analytics.models import TopicCluster

logger = logging.getLogger(__name__)


class TopicClusteringService:
    """
    AI-powered service to cluster questions by semantic meaning and analyze repetition patterns.
    Uses sentence-transformers for deep semantic understanding.
    Enhanced with HybridLLMService for improved similarity detection.
    """
    
    def __init__(
        self,
        subject: Subject,
        similarity_threshold: float = 0.55,  # Lower threshold for broader topic grouping
        tier_1_threshold: int = 4,  # Top Priority: 4+ times
        tier_2_threshold: int = 3,  # High Priority: 3 times
        tier_3_threshold: int = 2   # Medium Priority: 2 times
    ):
        self.subject = subject
        self.similarity_threshold = similarity_threshold
        self.same_question_threshold = 0.80  # same question band
        self.tier_1_threshold = tier_1_threshold
        self.tier_2_threshold = tier_2_threshold
        self.tier_3_threshold = tier_3_threshold
        
        # Initialize sentence transformer model if available
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading sentence transformer model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ AI model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load AI model: {e}, using fallback")
                self.model = None
        else:
            logger.info("Using enhanced keyword-based clustering (no AI model)")
        
        # Initialize HybridLLMService for improved similarity detection
        self.hybrid_llm = None
        try:
            from apps.analysis.services.hybrid_llm_service import HybridLLMService
            from django.conf import settings
            if settings.SIMILARITY_DETECTION.get('USE_HYBRID_APPROACH', False):
                self.hybrid_llm = HybridLLMService()
                logger.info("✅ HybridLLMService initialized for similarity detection")
        except Exception as e:
            logger.warning(f"Failed to initialize HybridLLMService: {e}, using embedding-only approach")
    
    def analyze_subject(self) -> Dict[str, Any]:
        """
        Main entry point: analyze all questions using AI semantic similarity.
        
        Returns:
            Statistics about the clustering process
        """
        logger.info(f"Starting AI-powered topic analysis for subject: {self.subject}")
        
        # Get all questions for this subject
        questions = Question.objects.filter(
            paper__subject=self.subject
        ).select_related('paper', 'module').order_by('module__number', 'question_number')
        
        if not questions.exists():
            logger.warning(f"No questions found for subject {self.subject}")
            return {'clusters_created': 0, 'questions_clustered': 0}
        
        # Clear existing clusters for this subject
        with transaction.atomic():
            TopicCluster.objects.filter(subject=self.subject).delete()
        
        # Group questions by module and cluster using AI
        modules = self.subject.modules.all()
        total_clusters = 0
        total_questions_clustered = 0
        
        for module in modules:
            module_questions = list(questions.filter(module=module))
            if module_questions:
                clusters_count, questions_count = self._cluster_module_questions_ai(module, module_questions)
                total_clusters += clusters_count
                total_questions_clustered += questions_count
                logger.info(f"Module {module.number}: Created {clusters_count} clusters from {len(module_questions)} questions")
        
        # Handle unclassified questions
        unclassified = list(questions.filter(module__isnull=True))
        if unclassified:
            clusters_count, questions_count = self._cluster_module_questions_ai(None, unclassified)
            total_clusters += clusters_count
            total_questions_clustered += questions_count
            logger.info(f"Unclassified: Created {clusters_count} clusters")
        
        logger.info(f"✅ Total: {total_clusters} clusters created, {total_questions_clustered} questions clustered")
        
        return {
            'clusters_created': total_clusters,
            'questions_clustered': total_questions_clustered
        }
    
    def _cluster_module_questions_ai(self, module: Optional[Module], questions: List[Question]) -> Tuple[int, int]:
        """
        Cluster questions within a module using AI or enhanced keyword matching.
        
        Args:
            module: The module (or None for unclassified)
            questions: List of Question objects
            
        Returns:
            Tuple of (clusters_created, questions_clustered)
        """
        if not questions:
            return 0, 0
        
        logger.info(f"Clustering {len(questions)} questions...")
        
        # Use AI if available, otherwise use enhanced keyword matching
        if self.model:
            return self._cluster_with_ai(module, questions)
        else:
            return self._cluster_with_keywords(module, questions)
    
    def _cluster_with_ai(self, module: Optional[Module], questions: List[Question]) -> Tuple[int, int]:
        """Cluster using sentence transformers (AI semantic similarity)."""
        # IMPORTANT: embed only normalized topic phrases, not raw OCR text
        key_terms = self._build_module_key_terms(questions)
        topic_phrases = [self._extract_topic_phrase(q.text, key_terms) for q in questions]

        logger.info("🤖 Generating AI embeddings...")
        embeddings = self.model.encode(topic_phrases, convert_to_tensor=False, show_progress_bar=False)

        clusters = []

        if SKLEARN_AVAILABLE and len(questions) > 1:
            try:
                dist_matrix = cosine_distances(embeddings)
                # distance_threshold = 1 - topic threshold (0.65)
                cluster_kwargs = {
                    'n_clusters': None,
                    'linkage': 'average',
                    'distance_threshold': 1 - self.similarity_threshold
                }
                try:
                    # Newer sklearn expects metric instead of affinity
                    clustering = AgglomerativeClustering(metric='precomputed', **cluster_kwargs)
                except TypeError:
                    # Older sklearn uses affinity
                    clustering = AgglomerativeClustering(affinity='precomputed', **cluster_kwargs)

                labels = clustering.fit_predict(dist_matrix)

                label_to_indices = defaultdict(list)
                for idx, label in enumerate(labels):
                    label_to_indices[label].append(idx)

                for idxs in label_to_indices.values():
                    cluster_questions = [questions[i] for i in idxs]
                    clusters.append({
                        'representative': cluster_questions[0],
                        'questions': cluster_questions
                    })
                logger.info(f"✅ Created {len(clusters)} clusters via Agglomerative")
                
                # Refine clusters using hybrid LLM if available
                if self.hybrid_llm:
                    clusters = self._refine_clusters_with_llm(clusters)
                    logger.info(f"✅ Refined to {len(clusters)} clusters using hybrid LLM")
                    
            except Exception as e:
                logger.warning(f"Agglomerative clustering failed, falling back to greedy: {e}")

        if not clusters:
            # Greedy fallback using cosine with thresholds + hybrid LLM for edge cases
            processed = set()
            for i, question in enumerate(questions):
                if i in processed:
                    continue
                cluster_indices = [i]
                processed.add(i)
                for j in range(len(questions)):
                    if j in processed or i == j:
                        continue
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    
                    # Use hybrid LLM service for edge cases if available
                    is_similar = False
                    if self.hybrid_llm and self.similarity_threshold <= similarity < self.same_question_threshold:
                        # Edge case: use LLM verification
                        try:
                            is_similar, conf, method, reason = self.hybrid_llm.are_questions_similar(
                                question.text,
                                questions[j].text,
                                question.marks if hasattr(question, 'marks') else None,
                                questions[j].marks if hasattr(questions[j], 'marks') else None
                            )
                            logger.debug(f"Hybrid check Q{i} vs Q{j}: {is_similar} ({conf:.2f}) via {method}")
                        except Exception as e:
                            logger.warning(f"Hybrid LLM similarity check failed: {e}")
                            is_similar = similarity >= self.similarity_threshold
                    else:
                        # Use threshold-based decision
                        is_similar = similarity >= self.similarity_threshold
                    
                    if is_similar:
                        cluster_indices.append(j)
                        processed.add(j)
                cluster_questions = [questions[idx] for idx in cluster_indices]
                clusters.append({
                    'representative': cluster_questions[0],
                    'questions': cluster_questions
                })
            logger.info(f"✅ Created {len(clusters)} clusters via greedy fallback with hybrid LLM verification")

        return self._save_clusters(module, clusters)
    
    def _cluster_with_keywords(self, module: Optional[Module], questions: List[Question]) -> Tuple[int, int]:
        """Enhanced keyword-based clustering (fallback when AI unavailable)."""
        logger.info("📊 Using enhanced keyword clustering...")

        key_terms = self._build_module_key_terms(questions)
        
        clusters = []
        processed = set()
        
        for i, question in enumerate(questions):
            if i in processed:
                continue
            
            cluster_indices = [i]
            processed.add(i)
            
            # Get keywords for this question
            topic_text_1 = self._extract_topic_phrase(question.text, key_terms)
            keywords1 = self._extract_keywords(topic_text_1)
            
            # Find similar questions by keyword overlap
            for j in range(len(questions)):
                if j in processed or i == j:
                    continue
                
                topic_text_2 = self._extract_topic_phrase(questions[j].text, key_terms)
                keywords2 = self._extract_keywords(topic_text_2)
                similarity = self._keyword_similarity(keywords1, keywords2)
                
                if similarity >= self.similarity_threshold:
                    cluster_indices.append(j)
                    processed.add(j)
            
            cluster_questions = [questions[idx] for idx in cluster_indices]
            clusters.append({
                'representative': cluster_questions[0],
                'questions': cluster_questions
            })
        
        logger.info(f"✅ Created {len(clusters)} keyword-based clusters")
        
        return self._save_clusters(module, clusters)
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity without numpy."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from already-normalized topic text."""
        text = self._normalize_question(text)
        
        # Remove common words
        stopwords = {
            'explain', 'describe', 'define', 'discuss', 'what', 'how', 'why', 
            'list', 'state', 'mention', 'briefly', 'detail', 'with', 'the', 
            'and', 'or', 'for', 'to', 'of', 'in', 'on', 'at', 'from'
        }
        
        words = text.split()
        keywords = {w for w in words if len(w) > 3 and w not in stopwords}
        
        return keywords
    
    def _keyword_similarity(self, keywords1: set, keywords2: set) -> float:
        """Calculate Jaccard similarity between keyword sets with boost for high overlap."""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        # Base Jaccard similarity
        base_similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if there's significant overlap (helps group reworded questions)
        overlap_ratio = len(intersection) / min(len(keywords1), len(keywords2)) if keywords1 and keywords2 else 0.0
        
        # If >60% of keywords overlap, boost the similarity
        if overlap_ratio > 0.6:
            return min(1.0, base_similarity * 1.3)
        
        return base_similarity
    
    def _refine_clusters_with_llm(self, clusters: List[Dict]) -> List[Dict]:
        """
        Refine clusters using hybrid LLM service to verify similarity.
        Splits clusters where questions are not actually similar according to LLM.
        
        Args:
            clusters: List of cluster dicts with 'representative' and 'questions' keys
            
        Returns:
            Refined list of clusters
        """
        if not self.hybrid_llm:
            return clusters
        
        refined_clusters = []
        total_splits = 0
        
        for cluster in clusters:
            questions = cluster['questions']
            
            # Single-question clusters don't need refinement
            if len(questions) <= 1:
                refined_clusters.append(cluster)
                continue
            
            # For multi-question clusters, verify each question is similar to the representative
            representative = questions[0]
            verified_questions = [representative]
            orphaned_questions = []
            
            for question in questions[1:]:
                try:
                    is_similar, conf, method, reason = self.hybrid_llm.are_questions_similar(
                        representative.text,
                        question.text,
                        representative.marks if hasattr(representative, 'marks') else None,
                        question.marks if hasattr(question, 'marks') else None
                    )
                    
                    if is_similar:
                        verified_questions.append(question)
                    else:
                        orphaned_questions.append(question)
                        logger.debug(f"Split question from cluster: {reason}")
                except Exception as e:
                    # On error, keep the question in the cluster (conservative)
                    logger.warning(f"LLM verification failed: {e}")
                    verified_questions.append(question)
            
            # Add the verified cluster
            if verified_questions:
                refined_clusters.append({
                    'representative': verified_questions[0],
                    'questions': verified_questions
                })
            
            # Create singleton clusters for orphaned questions
            for orphan in orphaned_questions:
                refined_clusters.append({
                    'representative': orphan,
                    'questions': [orphan]
                })
                total_splits += 1
        
        if total_splits > 0:
            logger.info(f"LLM refinement split {total_splits} questions into separate clusters")
        
        return refined_clusters
    
    def _save_clusters(self, module: Optional[Module], clusters: List[Dict]) -> Tuple[int, int]:
        """Save clusters to database."""
        created_count = 0
        questions_clustered = 0
        
        with transaction.atomic():
            for cluster_data in clusters:
                if len(cluster_data['questions']) > 0:
                    self._create_topic_cluster(module, cluster_data)
                    created_count += 1
                    questions_clustered += len(cluster_data['questions'])
        
        return created_count, questions_clustered
    
    def _normalize_question(self, text: str) -> str:
        """
        Normalize OCR/PDF text BEFORE embeddings.
        Goal: stable similarity even with noisy extraction.
        """
        if not text:
            return ''

        text = text.lower()

        # Drop marks, years, question numbering
        text = re.sub(r'\(\s*\d+\s*marks?\s*\)', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b20\d{2}\b', ' ', text)
        text = re.sub(r'^[q]\s*\d+[a-z]?\s*[:.)\-]?\s*', ' ', text, flags=re.IGNORECASE)

        # Replace all non-alnum with spaces (kills OCR garbage like € : ; | etc.)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Fix common OCR splits: "warm ing" -> "warming" (only when tail fragment is NOT a real word)
        protected = {
            'of', 'to', 'in', 'on', 'at', 'by', 'an', 'as', 'or', 'and', 'the', 'for', 'with',
        }

        def _join_ocr_split(match: re.Match) -> str:
            head = match.group(1)
            tail = match.group(2)
            if tail in protected:
                return f"{head} {tail}"
            return f"{head}{tail}"

        text = re.sub(r'\b([a-z]{3,})\s+([a-z]{1,3})\b', _join_ocr_split, text)

        # Remove filler exam verbs/phrases (do NOT affect display; only embeddings)
        remove_phrases = [
            'explain', 'define', 'discuss', 'describe', 'state', 'list', 'enumerate',
            'briefly', 'in detail', 'with diagram', 'with neat diagram', 'with example',
            'what is', 'what are', 'write short note on', 'write a note on',
        ]
        for phrase in remove_phrases:
            text = text.replace(phrase, ' ')

        # Collapse whitespace and drop single-letter tokens
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = [t for t in text.split() if len(t) > 1]
        return ' '.join(tokens)

    def _build_module_key_terms(self, questions: List[Question]) -> List[str]:
        """Derive common 2-4 word phrases within this module to stabilize topic-phrase extraction."""
        phrase_counts: Counter[str] = Counter()

        for q in questions:
            normalized = self._normalize_question(q.text)
            tokens = normalized.split()
            # Build n-grams (2..4) from meaningful tokens
            tokens = [t for t in tokens if len(t) > 2]
            for n in (2, 3, 4):
                for i in range(0, max(0, len(tokens) - n + 1)):
                    phrase = ' '.join(tokens[i:i + n])
                    phrase_counts[phrase] += 1

        # Keep phrases that appear at least twice (likely true topics)
        candidates = [p for p, c in phrase_counts.items() if c >= 2]
        candidates.sort(key=lambda p: (-phrase_counts[p], -len(p)))
        return candidates[:80]

    def _extract_topic_phrase(self, text: str, key_terms: List[str]) -> str:
        """
        Extract a topic phrase for embedding.
        - Prefer repeated phrases in this module (key_terms)
        - Fallback: first 5 words of normalized text
        """
        normalized = self._normalize_question(text)
        if not normalized:
            return ''

        # Prefer most frequent phrases in this module (key_terms are pre-sorted by frequency)
        for term in key_terms:
            if term and term in normalized:
                return self._strip_phrase_edges(term)

        return self._strip_phrase_edges(' '.join(normalized.split()[:5]))

    def _strip_phrase_edges(self, phrase: str) -> str:
        """Trim edge stopwords from a topic phrase (keeps the core concept stable)."""
        if not phrase:
            return ''

        stop = {'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'from', 'by', 'the', 'a', 'an'}
        parts = [p for p in phrase.split() if p]
        while parts and parts[0] in stop:
            parts.pop(0)
        while parts and parts[-1] in stop:
            parts.pop()
        return ' '.join(parts)
    
    def _are_similar(self, q1: Question, q2: Question) -> bool:
        """
        Determine if two questions are similar enough to be grouped.
        Uses text normalization and fuzzy matching.
        """
        # Normalize both texts
        norm1 = self._normalize_text(q1.text)
        norm2 = self._normalize_text(q2.text)
        
        # Check for very short texts (likely same topic)
        if len(norm1) < 30 or len(norm2) < 30:
            # Use simple substring matching for short questions
            return norm1 in norm2 or norm2 in norm1
        
        # Calculate simple similarity score
        similarity = self._calculate_text_similarity(norm1, norm2)
        
        return similarity >= self.similarity_threshold
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize question text for comparison.
        Removes marks, years, trivial words, and standardizes format.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove common patterns
        text = re.sub(r'\(\s*\d+\s*marks?\s*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{4}', '', text)  # Remove years
        text = re.sub(r'(dec|december|jun|june|nov|november|may|april|aug|august)\s*\d{4}', '', text, flags=re.IGNORECASE)
        
        # Remove question numbers and part indicators
        text = re.sub(r'^q\d+[a-z]?\s*[:\.\)]*\s*', '', text)
        text = re.sub(r'^question\s*\d+\s*[:\.\)]*\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^part\s*[ab]\s*[:\.\)]*\s*', '', text, flags=re.IGNORECASE)
        
        # Remove trivial words (but keep longer words even if in trivial list)
        trivial = ['the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in', 'on', 'at']
        words = text.split()
        words = [w for w in words if len(w) > 3 or w not in trivial]
        
        # Remove extra whitespace
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two normalized texts.
        Uses token-based Jaccard similarity.
        """
        # Tokenize
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_topic_name(self, question: Question) -> str:
        """
        Extract a CANONICAL topic name - core concept only, no action verbs.
        Multiple questions about the same topic MUST get the same name.
        """
        text = question.text
        
        # Remove marks notation first
        text = re.sub(r'\(\s*\d+\s*marks?\s*\)', '', text, flags=re.IGNORECASE)
        
        # Remove ALL action verbs (expanded list)
        action_verbs = r'^(explain|define|describe|discuss|what|how|why|list|enumerate|state|elaborate|illustrate|classify|compare|differentiate|contrast|distinguish|mention|identify|briefly|detail|outline|summarize|evaluate|analyze|analyse|examine|comment|write|give|draw|show|demonstrate|calculate|derive|prove|determine|find|compute|verify|check)\s+'
        text = re.sub(action_verbs, '', text, flags=re.IGNORECASE)
        
        # Remove filler phrases
        text = re.sub(r'\b(with\s+(the\s+)?(help\s+of\s+)?(a\s+)?(suitable\s+)?(neat\s+)?(labelled\s+)?(diagram|example|sketch|graph))\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(in\s+detail|briefly)\b', '', text, flags=re.IGNORECASE)
        
        # Take first sentence or clause
        if '.' in text[:100]:
            text = text.split('.')[0]
        elif '?' in text[:100]:
            text = text.split('?')[0]
        
        # Remove common articles and prepositions from start
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        
        # Truncate to core concept (max 60 chars)
        text = text[:60].strip()
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize properly - title case for main words
        words = text.split()
        capitalized = []
        for word in words:
            if len(word) > 3 or word.lower() in ['ktu', 'co2', 'ph', 'gis']:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        text = ' '.join(capitalized)
        
        return text if text else question.text[:50]
    
    def _create_topic_cluster(self, module: Optional[Module], cluster_data: Dict[str, Any]):
        """
        Create a TopicCluster object with proper priority calculation.
        """
        representative = cluster_data['representative']
        questions = cluster_data['questions']

        # Canonicalize topic name from topic phrases (NOT raw questions)
        module_key_terms = self._build_module_key_terms(questions)
        phrases = [self._extract_topic_phrase(q.text, module_key_terms) for q in questions]
        phrases = [p for p in phrases if p]
        canonical_phrase = Counter(phrases).most_common(1)[0][0] if phrases else self._normalize_question(representative.text)

        # Pretty name for reports (still derived from canonical phrase)
        topic_name = ' '.join(w.capitalize() if len(w) > 3 else w.lower() for w in canonical_phrase.split())
        
        # Calculate repetition statistics
        years = set()
        total_marks = 0
        part_a_count = 0
        part_b_count = 0
        
        for q in questions:
            # Get unique years
            if q.paper.year:
                years.add(str(q.paper.year))
            
            # Sum marks
            if q.marks:
                total_marks += q.marks

            # Part split is DISPLAY-ONLY; we still track distribution for info
            part = getattr(q, 'part', None)
            if part == 'A':
                part_a_count += 1
            elif part == 'B':
                part_b_count += 1
        
        # Frequency = number of unique years this topic appeared
        frequency_count = len(years)
        
        # Determine priority tier based on frequency (use integer values)
        if frequency_count >= self.tier_1_threshold:
            priority = 1  # TOP PRIORITY (4+ times)
        elif frequency_count >= self.tier_2_threshold:
            priority = 2  # HIGH PRIORITY (3 times)
        elif frequency_count >= self.tier_3_threshold:
            priority = 3  # MEDIUM PRIORITY (2 times)
        else:
            priority = 4  # LOW PRIORITY (1 time)
        
        # Create normalized key for deduplication
        normalized_key = canonical_phrase
        
        # Create TopicCluster
        cluster = TopicCluster.objects.create(
            subject=self.subject,
            module=module,
            topic_name=topic_name,
            normalized_key=normalized_key,
            representative_text=representative.text,
            frequency_count=frequency_count,
            years_appeared=sorted(list(years)),
            total_marks=total_marks,
            priority_tier=priority,
            question_count=len(questions),
            part_a_count=part_a_count,
            part_b_count=part_b_count,
            cluster_id=normalized_key[:100],
        )

        # Attach questions to this cluster (Part A/B is display-only; repetition uses cluster years)
        question_ids = [q.id for q in questions]
        Question.objects.filter(id__in=question_ids).update(
            topic_cluster=cluster,
            repetition_count=frequency_count,
            years_appeared=sorted(list(years)),
        )
        
        logger.debug(f"Created cluster: {topic_name} ({frequency_count} times, {priority})")


def analyze_subject_topics(
    subject: Subject,
    similarity_threshold: float = 0.55,  # Lower threshold for broader topic grouping
    tier_1_threshold: int = 4,  # TOP PRIORITY: 4+ times
    tier_2_threshold: int = 3,  # HIGH PRIORITY: 3 times
    tier_3_threshold: int = 2   # MEDIUM PRIORITY: 2 times
) -> Dict[str, Any]:
    """
    Convenience function to analyze topics for a subject.
    
    Args:
        subject: Subject instance
        similarity_threshold: Minimum similarity for clustering (0-1)
        tier_1_threshold: Minimum occurrences for Top Priority
        tier_2_threshold: Minimum occurrences for High Priority
        tier_3_threshold: Minimum occurrences for Medium Priority
    
    Returns:
        Statistics dictionary
    """
    service = TopicClusteringService(
        subject=subject,
        similarity_threshold=similarity_threshold,
        tier_1_threshold=tier_1_threshold,
        tier_2_threshold=tier_2_threshold,
        tier_3_threshold=tier_3_threshold
    )
    return service.analyze_subject()
