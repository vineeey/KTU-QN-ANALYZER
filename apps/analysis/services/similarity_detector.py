"""
Strict similarity detection for KTU repeated question analysis.

THRESHOLD POLICY (mandatory):
    < 0.80 → NOT same. NEVER cluster. No LLM call needed.
    0.80 – 0.82 → BORDERLINE. Treated as NOT same (below strict threshold).
    0.82 – 0.90 → Requires LLM confirmation (confidence ≥ 80 and SAME=YES).
    ≥ 0.90 → Requires LLM confirmation (confidence ≥ 80 and SAME=YES).

    A pair is ONLY marked as a strict repetition if:
        (1) cosine_similarity(embed(normalize(q1)), embed(normalize(q2))) >= 0.82
        AND
        (2) LLM returns SAME=YES with Confidence >= 80

TEXT NORMALIZATION before embedding:
    - Lowercase
    - Remove instructional verbs: explain, describe, discuss, define, write,
      list, state, derive, compare, differentiate, draw, illustrate, outline,
      elaborate, mention, calculate, solve
    - Remove leading question numbers and sub-question labels
    - Collapse whitespace

This ensures "Explain the Spiral model phases" and "Describe the phases of
the Spiral model" are correctly identified as SAME, while "Advantages of
Spiral model" vs "When is Spiral model used" are correctly identified as
DIFFERENT.
"""

from __future__ import annotations

import re
import logging
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────
STRICT_THRESHOLD = 0.82   # >= this → candidate for LLM confirmation
FLOOR_THRESHOLD = 0.80    # < this → NEVER cluster (hard reject)
# Between 0.80 and 0.82 is a dead zone — treated as NOT same.

# ── Instructional verbs to remove before embedding ────────────────────────
_INSTRUCTIONAL_VERBS = [
    "explain", "describe", "discuss", "define", "write", "list",
    "state", "derive", "compare", "differentiate", "draw", "illustrate",
    "outline", "elaborate", "mention", "calculate", "solve", "analyze",
    "analyse", "evaluate", "design", "implement", "show", "prove",
    "demonstrate", "identify", "what is", "what are", "give", "brief",
]

_RE_INSTRUCTIONAL = re.compile(
    r"^(?:" + "|".join(re.escape(v) for v in _INSTRUCTIONAL_VERBS) + r")\s+",
    re.IGNORECASE,
)

# ── Question number / sub-question prefix removal ─────────────────────────
_RE_Q_NUMBER = re.compile(r"^(?:q\.?\s*)?\d{1,2}\s*[.)]\s*", re.IGNORECASE)
_RE_SUB_LABEL = re.compile(
    r"^\(?(?:[a-z]|i{1,3}v?|vi{0,3}|ix|x)\)\.?\s+", re.IGNORECASE
)


def normalize_for_embedding(text: str) -> str:
    """
    Normalize question text before embedding to improve similarity accuracy.

    Steps:
    1. Strip question number/sub-question label prefix.
    2. Lowercase.
    3. Remove leading instructional verb (explain, describe, discuss, etc.).
    4. Collapse whitespace.
    5. Strip leading/trailing punctuation.

    Examples:
        "Explain the phases of the Spiral model."
            → "phases of the spiral model"
        "Describe phases of Spiral model"
            → "phases of spiral model"
        "Q1. a) Write about CMMI levels"
            → "cmmi levels"
    """
    # Step 1: remove question number
    text = _RE_Q_NUMBER.sub("", text).strip()
    # Step 2: remove sub-question label
    text = _RE_SUB_LABEL.sub("", text).strip()
    # Step 3: lowercase
    text = text.lower()
    # Step 4: remove leading instructional verb (one pass, greedy)
    # Apply repeatedly to catch "Explain and describe..." patterns
    for _ in range(3):
        new = _RE_INSTRUCTIONAL.sub("", text).strip()
        if new == text:
            break
        text = new
    # Step 5: collapse whitespace and strip punctuation
    text = re.sub(r"\s+", " ", text).strip().rstrip("?.,;:")
    return text


class SimilarityDetector:
    """
    Pairwise question similarity detector with strict threshold policy.

    Primary use: pre-filter before the LLM confirmation step in
    ClusteringService. Can also be used independently for ad-hoc checks.

    Usage::

        detector = SimilarityDetector()
        # Check one pair (returns full decision tuple):
        result = detector.is_repeated(q1_text, q2_text, embedding_model)
        # result: (is_same: bool, similarity: float, zone: str)
    """

    STRICT_THRESHOLD = STRICT_THRESHOLD
    FLOOR_THRESHOLD = FLOOR_THRESHOLD

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from django.conf import settings
            from sentence_transformers import SentenceTransformer
            model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
        return self._model

    def embed_normalized(self, text: str) -> np.ndarray:
        """
        Normalize *text* then compute its embedding vector.

        This is the canonical embedding for repetition detection.
        Raw (non-normalized) embeddings must NOT be used for this purpose.
        """
        normalized = normalize_for_embedding(text)
        model = self._get_model()
        vec = model.encode([normalized], show_progress_bar=False, convert_to_numpy=True)
        return vec[0]

    def compute_similarity(
        self, text1: str, text2: str
    ) -> Tuple[float, str]:
        """
        Compute normalized cosine similarity between two question texts.

        Returns:
            (similarity, zone) where zone is one of:
              'hard_reject'  – similarity < 0.80  → NEVER same
              'dead_zone'    – 0.80 <= sim < 0.82 → treated as NOT same
              'llm_required' – 0.82 <= sim < 0.90 → LLM confirmation needed
              'llm_required_high' – sim >= 0.90   → LLM confirmation needed
        """
        emb1 = self.embed_normalized(text1)
        emb2 = self.embed_normalized(text2)
        sim = float(cosine_similarity([emb1], [emb2])[0][0])
        zone = self._zone(sim)
        return sim, zone

    @staticmethod
    def _zone(similarity: float) -> str:
        if similarity >= 0.90:
            return "llm_required_high"
        if similarity >= STRICT_THRESHOLD:
            return "llm_required"
        if similarity >= FLOOR_THRESHOLD:
            return "dead_zone"
        return "hard_reject"

    def is_candidate_pair(self, text1: str, text2: str) -> Tuple[bool, float, str]:
        """
        Quick decision: should this pair proceed to LLM confirmation?

        Returns:
            (is_candidate, similarity, zone)
            - is_candidate=True means similarity >= 0.82, proceed to LLM.
            - is_candidate=False means reject without calling LLM.
        """
        sim, zone = self.compute_similarity(text1, text2)
        is_candidate = zone in ("llm_required", "llm_required_high")
        if not is_candidate:
            logger.debug(
                "Pair rejected (sim=%.3f, zone=%s). Not a candidate for repetition.",
                sim, zone,
            )
        return is_candidate, sim, zone

    def compute_similarity_matrix(
        self, texts: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise normalized similarity matrix for a list of questions.

        All vectors use normalized text (instructional verbs removed, etc.)

        Returns:
            (N, N) numpy array of cosine similarities.
        """
        normalized = [normalize_for_embedding(t) for t in texts]
        model = self._get_model()
        embeddings = model.encode(
            normalized, show_progress_bar=False, convert_to_numpy=True
        )
        return cosine_similarity(embeddings)

    def find_candidate_pairs(
        self, texts: List[str]
    ) -> List[Tuple[int, int, float, str]]:
        """
        Find all pairs that exceed the strict threshold (>= 0.82) and
        should proceed to LLM confirmation.

        Pairs below the floor (< 0.80) are silently rejected.
        Pairs in the dead zone (0.80–0.82) are also rejected.

        Returns:
            List of (i, j, similarity, zone) tuples, sorted by similarity desc.
        """
        sim_matrix = self.compute_similarity_matrix(texts)
        n = len(texts)
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i][j])
                zone = self._zone(sim)
                if zone in ("llm_required", "llm_required_high"):
                    candidates.append((i, j, sim, zone))
        candidates.sort(key=lambda x: x[2], reverse=True)
        logger.info(
            "Found %d candidate pairs (threshold=%.2f) out of %d total pairs.",
            len(candidates),
            STRICT_THRESHOLD,
            n * (n - 1) // 2,
        )
        return candidates
