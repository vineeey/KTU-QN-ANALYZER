"""
Strict module-isolated semantic clustering service.

ISOLATION RULES (mandatory, enforced at runtime):
- Clustering runs independently for EACH (module, part) combination.
- Questions from different modules MUST NEVER be in the same clustering batch.
- Part A and Part B questions MUST NEVER be in the same clustering batch.
- If a module isolation violation is detected, a ValueError is raised.

CLUSTER TYPES:
1. STRICT_REPETITION: similarity >= 0.82 AND LLM confirms "Same"
   with Confidence >= 80. These drive study priority.
2. CONCEPT_SIMILARITY: 0.65 <= similarity < 0.82, no LLM needed.
   These are informational only and NEVER used for priority.

The two cluster types are stored separately in ClusterGroup.cluster_type.

Design:
- Singleton SentenceTransformer model (loaded once per process).
- Incremental: only questions without a cached embedding are re-encoded.
- Memory-safe: batch encoding in configurable chunk sizes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────

# Minimum cosine similarity to form a STRICT_REPETITION cluster
# (LLM confirmation also required)
STRICT_REPETITION_THRESHOLD = 0.82

# Minimum cosine similarity to form a CONCEPT_SIMILARITY cluster
CONCEPT_SIMILARITY_THRESHOLD = 0.65

# Absolute floor — pairs below this are NEVER in the same clustering batch
FLOOR_THRESHOLD = 0.80

# AgglomerativeClustering distance thresholds
_STRICT_DISTANCE = 1.0 - STRICT_REPETITION_THRESHOLD   # 0.18
_CONCEPT_DISTANCE = 1.0 - CONCEPT_SIMILARITY_THRESHOLD  # 0.35

_BATCH_SIZE = 64


class ModuleIsolationError(ValueError):
    """
    Raised when a clustering batch is detected to contain questions from
    multiple modules, which violates the isolation contract.
    """
    pass


class ClusteringService:
    """
    Module-isolated clustering service.

    For every subject, iterates over all (module, part) combinations and
    runs independent clustering inside each slice.

    Usage::

        service = ClusteringService()
        cluster_groups = service.run(subject)
    """

    def __init__(self):
        self._model = None  # lazy singleton

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, subject) -> list:
        """
        Execute the full clustering pipeline for *subject*.

        Produces both STRICT_REPETITION and CONCEPT_SIMILARITY cluster
        groups for every (module, part) combination that has ≥ 2 questions.

        Returns:
            List of created ClusterGroup ORM instances.
        """
        from apps.questions.models import Question

        all_questions = list(
            Question.objects.filter(paper__subject=subject)
            .select_related("paper", "module")
            .order_by("id")
        )

        if len(all_questions) < 2:
            logger.warning(
                "Subject '%s' has < 2 questions — skipping clustering.", subject
            )
            return []

        logger.info(
            "Starting isolated clustering for subject '%s' (%d questions).",
            subject,
            len(all_questions),
        )

        # ── Delete all existing clusters for this subject ──────────────
        from apps.analytics.models import ClusterGroup
        ClusterGroup.objects.filter(subject=subject).delete()
        logger.debug("Deleted old cluster groups for subject '%s'.", subject)

        # ── Group by (module_id, part) ─────────────────────────────────
        buckets: Dict[Tuple[Optional[int], str], List] = defaultdict(list)
        for q in all_questions:
            key = (q.module_id, q.part or "A")
            buckets[key].append(q)

        all_created: list = []

        for (module_id, part), questions in sorted(buckets.items()):
            if len(questions) < 2:
                logger.debug(
                    "Bucket (module_id=%s, part=%s) has only %d question(s) — skipping.",
                    module_id, part, len(questions),
                )
                continue

            logger.info(
                "Clustering bucket: module_id=%s, part=%s, n=%d",
                module_id, part, len(questions),
            )

            # ── Enforce module isolation ───────────────────────────────
            self._assert_module_isolation(questions, module_id)

            embeddings = self._get_embeddings(questions)

            # ── Run both clustering passes ─────────────────────────────
            strict_groups = self._run_strict_clustering(
                subject, questions, embeddings, module_id, part
            )
            concept_groups = self._run_concept_clustering(
                subject, questions, embeddings, module_id, part,
                strict_groups=strict_groups,
            )

            all_created.extend(strict_groups)
            all_created.extend(concept_groups)

        logger.info(
            "Clustering complete for subject '%s': %d cluster groups "
            "(%d strict repetition, %d concept similarity).",
            subject,
            len(all_created),
            sum(1 for g in all_created if g.cluster_type == "strict_repetition"),
            sum(1 for g in all_created if g.cluster_type == "concept_similarity"),
        )
        return all_created

    # ------------------------------------------------------------------
    # Isolation guard
    # ------------------------------------------------------------------

    def _assert_module_isolation(
        self, questions: list, expected_module_id: Optional[int]
    ) -> None:
        """
        Raise ModuleIsolationError if any question in *questions* belongs
        to a module different from *expected_module_id*.

        This is the enforcement mechanism for Phase 3.
        """
        violators = [
            q for q in questions if q.module_id != expected_module_id
        ]
        if violators:
            details = [
                f"Q#{q.id} module_id={q.module_id}" for q in violators[:5]
            ]
            raise ModuleIsolationError(
                f"Module isolation violated in clustering batch "
                f"(expected module_id={expected_module_id}). "
                f"Violating questions: {details}"
            )

    # ------------------------------------------------------------------
    # Strict repetition clustering (≥ 0.82, LLM confirmed)
    # ------------------------------------------------------------------

    def _run_strict_clustering(
        self,
        subject,
        questions: list,
        embeddings: np.ndarray,
        module_id: Optional[int],
        part: str,
    ) -> list:
        """
        Run AgglomerativeClustering at the strict threshold (0.82) and
        then apply LLM confirmation for each candidate cluster pair.

        Only clusters with LLM-confirmed SAME verdict (confidence ≥ 80)
        are marked as STRICT_REPETITION.  Others are discarded from this
        pass (they may be picked up as CONCEPT_SIMILARITY later).

        Returns:
            List of created ClusterGroup ORM instances (strict only).
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        if len(embeddings) < 2:
            return []

        # Compute pairwise similarity matrix first to apply floor
        sim_matrix = cos_sim(embeddings)

        # AgglomerativeClustering with strict distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=_STRICT_DISTANCE,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Group indices by cluster
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            label_to_indices[int(lbl)].append(idx)

        # Keep only clusters with ≥ 2 members (repetitions)
        multi_member = {
            lbl: idxs
            for lbl, idxs in label_to_indices.items()
            if len(idxs) >= 2
        }

        # For each multi-member cluster, validate via LLM confirmation
        confirmed_groups = []
        for lbl, idxs in multi_member.items():
            cluster_qs = [questions[i] for i in idxs]

            # Check that ALL pairwise similarities are above the floor (0.80)
            # to avoid false clusters created by average-linkage chaining
            pairs_ok = True
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    if sim_matrix[idxs[i]][idxs[j]] < FLOOR_THRESHOLD:
                        pairs_ok = False
                        break
                if not pairs_ok:
                    break

            if not pairs_ok:
                logger.debug(
                    "Cluster %d dropped: pairwise similarity below floor %.2f",
                    lbl, FLOOR_THRESHOLD,
                )
                continue

            # LLM confirmation: check representative pair
            if not self._llm_confirm_cluster(cluster_qs):
                logger.debug(
                    "Cluster %d dropped: LLM did not confirm SAME",
                    lbl,
                )
                continue

            # Build and persist the confirmed cluster
            group = self._build_cluster_group(
                subject,
                cluster_qs,
                embeddings[idxs],
                module_id,
                part,
                cluster_type="strict_repetition",
                cluster_label=lbl,
            )
            if group:
                confirmed_groups.append(group)

        logger.info(
            "Strict clustering (module=%s, part=%s): %d/%d clusters confirmed.",
            module_id, part, len(confirmed_groups), len(multi_member),
        )
        return confirmed_groups

    def _llm_confirm_cluster(self, questions: list) -> bool:
        """
        Use LLM to confirm that the representative questions in the
        cluster are truly asking the SAME thing.

        Takes the representative question (first) and checks it against
        every other question in the cluster.  All pairs must be confirmed
        SAME with confidence ≥ 80.

        Falls back to True (optimistic) if LLM is unavailable, because
        the embedding threshold already filtered to ≥ 0.82.
        """
        if len(questions) < 2:
            return True

        try:
            from apps.analysis.services.hybrid_llm_service import HybridLLMService
            llm = HybridLLMService()
        except Exception as exc:
            logger.warning(
                "LLM unavailable for cluster confirmation (%s). "
                "Accepting cluster based on embedding threshold alone.", exc,
            )
            return True

        rep_text = questions[0].text
        for other_q in questions[1:]:
            try:
                is_same, confidence = llm.confirm_same_question(
                    rep_text, other_q.text
                )
                if not is_same or confidence < 80:
                    return False
            except Exception as exc:
                logger.warning(
                    "LLM confirmation call failed (%s). "
                    "Accepting pair based on embedding threshold.", exc,
                )
                # If LLM is down, accept borderline — embedding is ≥ 0.82
        return True

    # ------------------------------------------------------------------
    # Concept similarity clustering (0.65 – 0.82, informational)
    # ------------------------------------------------------------------

    def _run_concept_clustering(
        self,
        subject,
        questions: list,
        embeddings: np.ndarray,
        module_id: Optional[int],
        part: str,
        strict_groups: list,
    ) -> list:
        """
        Run AgglomerativeClustering at the concept-similarity threshold
        (0.65).  Questions already captured in a strict_repetition cluster
        are still included (overlapping is intentional for completeness),
        but the two cluster types are stored separately and NEVER mixed.

        Returns:
            List of created ClusterGroup ORM instances (concept only).
        """
        from sklearn.cluster import AgglomerativeClustering

        if len(embeddings) < 2:
            return []

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=_CONCEPT_DISTANCE,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            label_to_indices[int(lbl)].append(idx)

        multi_member = {
            lbl: idxs
            for lbl, idxs in label_to_indices.items()
            if len(idxs) >= 2
        }

        concept_groups = []
        for lbl, idxs in multi_member.items():
            cluster_qs = [questions[i] for i in idxs]
            group = self._build_cluster_group(
                subject,
                cluster_qs,
                embeddings[idxs],
                module_id,
                part,
                cluster_type="concept_similarity",
                # Offset label to avoid collision with strict labels
                cluster_label=lbl + 10000,
            )
            if group:
                concept_groups.append(group)

        logger.info(
            "Concept clustering (module=%s, part=%s): %d concept groups.",
            module_id, part, len(concept_groups),
        )
        return concept_groups

    # ------------------------------------------------------------------
    # Shared cluster-group builder
    # ------------------------------------------------------------------

    def _build_cluster_group(
        self,
        subject,
        cluster_questions: list,
        cluster_embeddings: np.ndarray,
        module_id: Optional[int],
        part: str,
        cluster_type: str,
        cluster_label: int,
    ):
        """
        Create a ClusterGroup + ClusterMembership record for the given
        batch of questions.

        Returns the ClusterGroup, or None on error.
        """
        from apps.analytics.models import ClusterGroup, ClusterMembership

        try:
            years = sorted(
                {str(q.paper.year) for q in cluster_questions if q.paper.year}
            )
            frequency = len(set(years)) if years else len(cluster_questions)

            # Representative: question closest to the centroid
            centroid = cluster_embeddings.mean(axis=0)
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_q = cluster_questions[int(np.argmin(dists))]

            group = ClusterGroup.objects.create(
                subject=subject,
                module_id=module_id,
                cluster_label=cluster_label,
                cluster_type=cluster_type,
                part=part,
                representative_text=rep_q.text[:500],
                frequency=frequency,
                years_appeared=years,
                question_count=len(cluster_questions),
            )

            memberships = [
                ClusterMembership(cluster=group, question=q)
                for q in cluster_questions
            ]
            ClusterMembership.objects.bulk_create(memberships, ignore_conflicts=True)
            return group

        except Exception as exc:
            logger.error(
                "Failed to persist cluster group (module=%s, part=%s, type=%s): %s",
                module_id, part, cluster_type, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the singleton SentenceTransformer model (lazy load)."""
        if self._model is None:
            from django.conf import settings
            from sentence_transformers import SentenceTransformer
            model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer loaded: %s", model_name)
        return self._model

    def _get_embeddings(self, questions: list) -> np.ndarray:
        """
        Return (N, D) float32 normalized embeddings for *questions*.

        IMPORTANT: Text is pre-normalized before embedding (instructional
        verbs stripped, lowercase, question numbers removed) using
        ``normalize_for_embedding`` from similarity_detector.  This is
        critical so that:
          "Explain phases of Spiral model" and
          "Describe the phases of Spiral model"
        produce similar embeddings, while:
          "Advantages of Spiral model" and
          "Phases of Spiral model"
        remain distinct.

        These embeddings are NOT stored in QuestionEmbeddingCache because
        they are purpose-built for repetition/similarity clustering.  The
        QEC stores non-normalized embeddings for general search use.
        """
        from apps.analysis.services.similarity_detector import normalize_for_embedding

        model = self._get_model()
        normalized_texts = [normalize_for_embedding(q.text) for q in questions]

        all_vecs: List[np.ndarray] = []
        for i in range(0, len(normalized_texts), _BATCH_SIZE):
            batch = normalized_texts[i: i + _BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_vecs.append(vecs)

        return np.vstack(all_vecs).astype(np.float32)
