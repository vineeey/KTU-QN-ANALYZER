"""
Background tasks for the full 10-step analysis pipeline using Django-Q2.

Pipeline steps per paper:
  1.  PDF → page images (300 DPI) via PyMuPDF
  2.  Image preprocessing  (OpenCV)
  3.  OCR                  (PaddleOCR)
  4.  Text cleaning        (text_cleaner)
  5.  Question segmentation (Segmenter)
  6.  Module classification (two-stage: rule-based + semantic)
  7.  Embedding generation  (sentence-transformers, cached)
  8.  Semantic clustering   (AgglomerativeClustering)
  9.  Priority assignment   (PriorityEngine)
  10. Analytics update      (TopicCluster / ClusterGroup counters)

Steps 8–10 are run at the subject level (triggered after all papers
for a subject are processed, or on demand via analyze_subject_topics_task).
"""
import logging

from django_q.tasks import async_task

from apps.papers.models import Paper
from apps.subjects.models import Subject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _update_status(paper: Paper, status: str, detail: str = "", pct: int = 0):
    Paper.objects.filter(pk=paper.pk).update(
        status=status,
        status_detail=detail,
        progress_percent=pct,
    )


# ---------------------------------------------------------------------------
# Main paper processing task (steps 1–7)
# ---------------------------------------------------------------------------


def analyze_paper_task(paper_id: str):
    """
    Background task: run the full OCR → segmentation → classification →
    embedding pipeline for a single paper.

    Enqueued by :func:`queue_paper_analysis`.
    """
    try:
        paper = Paper.objects.select_related("subject").get(id=paper_id)
    except Paper.DoesNotExist:
        logger.error("analyze_paper_task: Paper %s does not exist.", paper_id)
        return

    _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Starting pipeline…", 5)

    try:
        # ----------------------------------------------------------------
        # Step 1 – PDF → images
        # ----------------------------------------------------------------
        _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Rendering PDF pages…", 10)
        from apps.analysis.services.ocr_engine import OCREngine
        ocr_engine = OCREngine()
        ocr_results = ocr_engine.process_pdf(paper.file.path)

        if not ocr_results:
            raise ValueError("PDF rendering produced no pages.")

        # ----------------------------------------------------------------
        # Steps 2–3 are handled inside OCREngine.process_pdf (preprocessing
        # + PaddleOCR). Raw text per page is in ocr_results[i].text.
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Step 4 – Text cleaning
        # ----------------------------------------------------------------
        _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Cleaning OCR text…", 30)
        from apps.analysis.services.text_cleaner import TextCleaner
        cleaner = TextCleaner()

        # Attempt LLM-based contextual cleaning if hybrid LLM is configured
        try:
            from apps.analysis.services.hybrid_llm_service import HybridLLMService
            llm = HybridLLMService()
            llm_fn = llm.clean_ocr_text if hasattr(llm, "clean_ocr_text") else None
        except Exception:
            llm_fn = None

        full_text_parts = [cleaner.clean(r.text, llm_fn=llm_fn) for r in ocr_results]
        full_text = "\n\n".join(full_text_parts)

        # Persist raw text on the paper record for debugging
        Paper.objects.filter(pk=paper.pk).update(raw_text=full_text[:50000])

        # ----------------------------------------------------------------
        # Step 5 – Question segmentation
        # ----------------------------------------------------------------
        _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Segmenting questions…", 45)
        from apps.analysis.services.segmenter import Segmenter
        segmenter = Segmenter()
        dtos = segmenter.parse(full_text)

        if not dtos:
            logger.warning("No questions extracted from paper '%s'.", paper)
            _update_status(paper, Paper.ProcessingStatus.COMPLETED, "No questions found.", 100)
            Paper.objects.filter(pk=paper.pk).update(
                status=Paper.ProcessingStatus.COMPLETED,
                questions_extracted=0,
                progress_percent=100,
            )
            return

        # ----------------------------------------------------------------
        # Step 6 – Module classification (strict heading-first)
        # ----------------------------------------------------------------
        _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Classifying modules…", 60)
        from apps.analysis.services.classifier import ModuleClassifier
        from apps.subjects.models import Module

        modules = list(Module.objects.filter(subject=paper.subject))
        module_map = {m.number: m for m in modules}
        classifier = ModuleClassifier()

        questions_created = []
        from apps.questions.models import Question
        for dto in dtos:
            if not dto.is_valid():
                continue

            # ── Module assignment rules (Phase 2 / Phase 3) ──────────────
            # Rule 1: If the segmenter found an explicit module heading in
            #         the paper, that assignment is FINAL.  The classifier
            #         must NEVER override an explicit heading.
            # Rule 2: Only call the classifier when no explicit heading was
            #         found (module_hint_is_explicit is False and module_hint
            #         is None).
            if dto.module_hint_is_explicit and dto.module_hint is not None:
                # Explicit paper heading — use it directly
                module_instance = module_map.get(dto.module_hint)
            elif not dto.module_hint_is_explicit and dto.module_hint is None:
                # No heading found → use classifier as fallback
                mod_num = classifier.classify(
                    dto.full_question_text,
                    paper.subject,
                    modules,
                    module_hint=None,
                )
                module_instance = module_map.get(mod_num) if mod_num else None
            else:
                # module_hint set but not explicitly — use it, allow override
                mod_num = dto.module_hint
                module_instance = module_map.get(mod_num)

            # ── Determine year ────────────────────────────────────────────
            years_appeared = []
            if dto.year_context:
                years_appeared = [dto.year_context]
            elif paper.year:
                years_appeared = [str(paper.year)]

            # ── Create / update Question record ──────────────────────────
            # Use full_question_text (text + sub-questions) for embedding.
            # This is the canonical text stored in Question.text.
            q, created = Question.objects.get_or_create(
                paper=paper,
                question_number=dto.question_number,
                defaults={
                    "text": dto.full_question_text,
                    "sub_questions": dto.sub_questions,
                    "marks": dto.marks,
                    "part": dto.part,
                    "module": module_instance,
                    "years_appeared": years_appeared,
                    # Track whether module assignment came from explicit heading
                    "module_manually_set": dto.module_hint_is_explicit,
                },
            )
            if not created:
                # Update fields that may have changed on re-runs
                update_fields = []
                if module_instance and q.module is None:
                    q.module = module_instance
                    q.module_manually_set = dto.module_hint_is_explicit
                    update_fields.extend(["module", "module_manually_set"])
                if dto.part and not q.part:
                    q.part = dto.part
                    update_fields.append("part")
                if not q.years_appeared and years_appeared:
                    q.years_appeared = years_appeared
                    update_fields.append("years_appeared")
                if update_fields:
                    q.save(update_fields=update_fields, skip_validation=True)
            questions_created.append(q)

        Paper.objects.filter(pk=paper.pk).update(
            questions_extracted=len(questions_created),
            questions_classified=sum(1 for q in questions_created if q.module_id),
        )

        # ----------------------------------------------------------------
        # Step 7 – Embedding generation (cached)
        # ----------------------------------------------------------------
        _update_status(paper, Paper.ProcessingStatus.PROCESSING, "Generating embeddings…", 80)
        try:
            from apps.analysis.services.embedder import EmbeddingService
            from apps.questions.models import QuestionEmbeddingCache
            import numpy as np

            embedder = EmbeddingService()
            uncached = [
                q for q in questions_created
                if not QuestionEmbeddingCache.objects.filter(question=q).exists()
            ]
            if uncached:
                texts = [q.text for q in uncached]
                vecs = embedder.encode_batch(texts)
                for q, vec in zip(uncached, vecs):
                    QuestionEmbeddingCache.objects.update_or_create(
                        question=q,
                        defaults={"vector": vec.tolist() if hasattr(vec, "tolist") else list(vec)},
                    )
            logger.info("Embeddings cached for %d questions.", len(uncached))
        except Exception as emb_exc:
            logger.warning("Embedding step skipped: %s", emb_exc)

        # ----------------------------------------------------------------
        # Mark paper complete
        # ----------------------------------------------------------------
        from django.utils import timezone
        Paper.objects.filter(pk=paper.pk).update(
            status=Paper.ProcessingStatus.COMPLETED,
            processed_at=timezone.now(),
            progress_percent=100,
            status_detail="Completed",
        )
        logger.info("Paper '%s' processed: %d questions.", paper, len(questions_created))

    except Exception as exc:
        logger.exception("Pipeline failed for paper %s: %s", paper_id, exc)
        Paper.objects.filter(pk=paper_id).update(
            status=Paper.ProcessingStatus.FAILED,
            processing_error=str(exc),
            progress_percent=0,
        )


# ---------------------------------------------------------------------------
# Subject-level task (steps 8–10)
# ---------------------------------------------------------------------------


def analyze_subject_topics_task(subject_id: str):
    """
    Background task: run clustering + priority assignment for all questions
    belonging to *subject_id*.

    Steps:
        8.  Semantic clustering   (AgglomerativeClustering)
        9.  Priority assignment   (PriorityEngine)
        10. Analytics counters update
    """
    try:
        subject = Subject.objects.get(id=subject_id)
    except Subject.DoesNotExist:
        logger.error("analyze_subject_topics_task: Subject %s not found.", subject_id)
        return

    try:
        # Step 8 – Clustering
        logger.info("Step 8: Clustering questions for subject '%s'.", subject)
        from apps.analytics.services.clustering import ClusteringService
        clustering_service = ClusteringService()
        cluster_groups = clustering_service.run(subject)

        # Step 9 – Priority assignment
        logger.info("Step 9: Assigning priorities for subject '%s'.", subject)
        from apps.analytics.services.priority_engine import PriorityEngine
        priority_engine = PriorityEngine()
        priority_engine.assign_for_subject(subject)

        # Step 10 – Analytics update (legacy TopicCluster compatibility)
        logger.info("Step 10: Updating analytics counters for subject '%s'.", subject)
        try:
            from apps.analytics.clustering import analyze_subject_topics
            analyze_subject_topics(subject)
        except Exception as legacy_exc:
            logger.debug("Legacy analytics update skipped: %s", legacy_exc)

        logger.info(
            "Subject '%s' topic analysis complete: %d cluster groups.",
            subject, len(cluster_groups),
        )
        return {"cluster_groups": len(cluster_groups)}

    except Exception as exc:
        logger.exception("Topic analysis failed for subject %s: %s", subject_id, exc)


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------


def queue_paper_analysis(paper: Paper):
    """Enqueue a paper for background OCR + extraction processing."""
    async_task(
        "apps.analysis.tasks.analyze_paper_task",
        str(paper.id),
        task_name=f"analyze_paper_{paper.id}",
    )
    logger.info("Queued paper analysis for paper '%s'.", paper)


def queue_topic_analysis(subject: Subject):
    """Enqueue subject-level clustering + priority assignment."""
    async_task(
        "apps.analysis.tasks.analyze_subject_topics_task",
        str(subject.id),
        task_name=f"analyze_topics_{subject.id}",
    )
    logger.info("Queued topic analysis for subject '%s'.", subject)
