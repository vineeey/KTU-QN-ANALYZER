"""
Two-stage module classifier.

Stage 1 – Deterministic rule-based mapping using module keywords / topics.
Stage 2 – Semantic fallback using sentence-transformers cosine similarity
          (singleton model loaded once per process).

Embeddings for module descriptions are computed once per classifier
instance and reused across calls.  Question embeddings are batch-encoded.
"""
import logging
import re
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class ModuleClassifier:
    """
    Classifies questions into modules using a two-stage approach.

    Stage 1: Keyword / topic matching (fast, deterministic, zero-cost).
    Stage 2: Semantic cosine similarity via sentence-transformers.

    The sentence-transformer model is loaded as a **class-level singleton**
    so it is shared across all instances and never loaded more than once
    per process.

    Usage::

        classifier = ModuleClassifier()
        module_num = classifier.classify(question_text, subject, modules)

        # Batch (more efficient):
        nums = classifier.classify_batch(texts, subject, modules)
    """

    # ----------------------------------------------------------------
    # Class-level singleton for the embedding model
    # ----------------------------------------------------------------
    _st_model = None
    _st_model_name: str = ""
    _st_loaded: bool = False

    # ----------------------------------------------------------------
    # Class-level module description cache
    # key: subject_id → {"module_num": (Module, embedding)}
    # ----------------------------------------------------------------
    _module_embedding_cache: Dict = {}

    # ----------------------------------------------------------------
    # Constants
    # ----------------------------------------------------------------
    _KEYWORD_SCORE_THRESHOLD: int = 1   # minimum keyword score to accept Stage 1
    _SEMANTIC_SIMILARITY_THRESHOLD: float = 0.30  # minimum cosine sim for Stage 2

    # LLM prompts kept for optional third-stage LLM use
    CLASSIFY_PROMPT = """
You are a question classifier for academic subjects. Given a question and the available modules,
determine which module the question belongs to.

Subject: {subject_name}
Modules:
{modules_list}

Question: {question_text}

Respond with ONLY the module number (e.g., "1", "2", "3"). If unsure, respond with "0".
"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._ensure_model_loaded()

    # ----------------------------------------------------------------
    # Singleton loader
    # ----------------------------------------------------------------

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """Load SentenceTransformer once (lazy, class-level singleton)."""
        if cls._st_loaded:
            return
        try:
            from django.conf import settings
            from sentence_transformers import SentenceTransformer

            model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            cls._st_model = SentenceTransformer(model_name)
            cls._st_model_name = model_name
            cls._st_loaded = True
            logger.info("✓ ModuleClassifier: SentenceTransformer loaded (%s).", model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed – semantic fallback disabled.")
        except Exception as exc:
            logger.warning("Could not load SentenceTransformer: %s", exc)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def classify(
        self,
        question_text: str,
        subject,
        modules: List,
        module_hint: Optional[int] = None,
    ) -> Optional[int]:
        """
        Classify a single question into a module number.

        Args:
            question_text: The question text to classify.
            subject: Subject ORM instance (used for logging / LLM prompt).
            modules: List of Module ORM instances.
            module_hint: Explicit module number from the PDF header (highest priority).

        Returns:
            Module number (int) or None if classification fails.
        """
        if not modules:
            return None

        # Priority 0 – explicit PDF header hint
        if module_hint is not None:
            if any(m.number == module_hint for m in modules):
                return module_hint

        # Stage 1 – keyword / topic matching
        result = self._stage1_keywords(question_text, modules)
        if result is not None:
            return result

        # Stage 2 – semantic cosine similarity
        result = self._stage2_semantic(question_text, subject, modules)
        if result is not None:
            return result

        # Stage 3 – optional LLM (only if client is configured)
        if self.llm_client:
            result = self._stage3_llm(question_text, subject, modules)
            if result is not None:
                return result

        return None

    def classify_batch(
        self,
        questions: List[str],
        subject,
        modules: List,
    ) -> List[Optional[int]]:
        """
        Classify a batch of questions efficiently.

        Stage 1 is applied per-question.  Stage 2 encodes all unmatched
        questions in a single batch call to the sentence-transformer model.

        Args:
            questions: List of question text strings.
            subject: Subject ORM instance.
            modules: List of Module ORM instances.

        Returns:
            List of module numbers aligned with *questions*.
        """
        if not modules or not questions:
            return [None] * len(questions)

        results: List[Optional[int]] = [None] * len(questions)
        unmatched_indices: List[int] = []

        # Stage 1
        for i, text in enumerate(questions):
            r = self._stage1_keywords(text, modules)
            if r is not None:
                results[i] = r
            else:
                unmatched_indices.append(i)

        if not unmatched_indices:
            return results

        # Stage 2 – batch encode unmatched questions
        if self.__class__._st_loaded and self.__class__._st_model:
            try:
                import numpy as np

                unmatched_texts = [questions[i] for i in unmatched_indices]
                module_descs, module_embs = self._get_module_embeddings(subject, modules)

                q_embs = self.__class__._st_model.encode(
                    unmatched_texts, show_progress_bar=False, convert_to_numpy=True
                )

                # Cosine similarity: (N_q, D) × (D, N_m) → (N_q, N_m)
                q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
                m_norm = module_embs / (np.linalg.norm(module_embs, axis=1, keepdims=True) + 1e-9)
                sims = q_norm @ m_norm.T  # (N_q, N_m)

                for row_idx, q_idx in enumerate(unmatched_indices):
                    best_col = int(np.argmax(sims[row_idx]))
                    best_sim = float(sims[row_idx, best_col])
                    if best_sim >= self._SEMANTIC_SIMILARITY_THRESHOLD:
                        results[q_idx] = module_descs[best_col]
            except Exception as exc:
                logger.warning("Stage 2 batch semantic classification failed: %s", exc)

        return results

    # ----------------------------------------------------------------
    # Stage 1 – keyword matching
    # ----------------------------------------------------------------

    def _stage1_keywords(self, question_text: str, modules: List) -> Optional[int]:
        """Return the module number with highest keyword/topic match score."""
        question_lower = question_text.lower()
        best_module: Optional[int] = None
        best_score: int = 0

        for module in modules:
            score = 0
            keywords: List[str] = list(module.keywords) if module.keywords else []
            topics: List[str] = list(module.topics) if module.topics else []

            for kw in keywords:
                if kw.lower() in question_lower:
                    score += 3

            for topic in topics:
                if topic.lower() in question_lower:
                    score += 2

            # Module name words
            if module.name:
                for word in module.name.lower().split():
                    if len(word) > 3 and word in question_lower:
                        score += 1

            if score > best_score:
                best_score = score
                best_module = module.number

        return best_module if best_score >= self._KEYWORD_SCORE_THRESHOLD else None

    # ----------------------------------------------------------------
    # Stage 2 – semantic similarity
    # ----------------------------------------------------------------

    def _get_module_embeddings(self, subject, modules: List):
        """
        Return (module_number_list, embedding_matrix) for *modules*.

        Results are cached per subject to avoid redundant encoding.
        """
        import numpy as np

        subject_key = getattr(subject, "id", id(subject))
        cache = self.__class__._module_embedding_cache

        if subject_key not in cache:
            descriptions = [
                f"{m.name}: {' '.join(m.topics[:5]) if m.topics else ''} "
                f"{' '.join(m.keywords[:5]) if m.keywords else ''}"
                for m in modules
            ]
            embs = self.__class__._st_model.encode(
                descriptions, show_progress_bar=False, convert_to_numpy=True
            )
            cache[subject_key] = {
                "module_numbers": [m.number for m in modules],
                "embeddings": embs,
            }
            logger.debug("Cached module embeddings for subject '%s'.", subject)

        entry = cache[subject_key]
        return entry["module_numbers"], entry["embeddings"]

    def _stage2_semantic(
        self, question_text: str, subject, modules: List
    ) -> Optional[int]:
        """Single-question semantic classification (slow path)."""
        if not self.__class__._st_loaded or not self.__class__._st_model:
            return None
        try:
            import numpy as np

            module_nums, module_embs = self._get_module_embeddings(subject, modules)
            q_emb = self.__class__._st_model.encode(
                [question_text], show_progress_bar=False, convert_to_numpy=True
            )
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            m_norm = module_embs / (np.linalg.norm(module_embs, axis=1, keepdims=True) + 1e-9)
            sims = (q_norm @ m_norm.T).flatten()
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self._SEMANTIC_SIMILARITY_THRESHOLD:
                return module_nums[best_idx]
        except Exception as exc:
            logger.warning("Stage 2 semantic classification failed: %s", exc)
        return None

    # ----------------------------------------------------------------
    # Stage 3 – optional LLM
    # ----------------------------------------------------------------

    def _stage3_llm(
        self, question_text: str, subject, modules: List
    ) -> Optional[int]:
        """Classify using LLM as a last resort."""
        modules_text = "\n".join(
            f"{m.number}. {m.name}: {', '.join(list(m.topics)[:5]) if m.topics else 'N/A'}"
            for m in modules
        )
        prompt = self.CLASSIFY_PROMPT.format(
            subject_name=subject.name,
            modules_list=modules_text,
            question_text=question_text[:500],
        )
        try:
            response = self.llm_client.generate(prompt, max_tokens=10).strip()
            match = re.search(r"\d+", response)
            if match:
                num = int(match.group())
                if any(m.number == num for m in modules):
                    return num
        except Exception as exc:
            logger.error("Stage 3 LLM classification failed: %s", exc)
        return None

    # ----------------------------------------------------------------
    # Legacy compatibility shim
    # ----------------------------------------------------------------

    def classify_by_keywords(self, question_text: str, modules: List) -> Optional[int]:
        """Alias kept for backward compatibility with the old pipeline."""
        return self._stage1_keywords(question_text, modules)

