"""
Similarity detection using cosine similarity.
Detects repeated/similar questions across years.
Enhanced with hybrid LLM approach for edge cases.
"""
import logging
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from django.conf import settings

logger = logging.getLogger(__name__)


class SimilarityDetector:
    """
    Detects similar/repeated questions using hybrid approach:
    - Fast embedding-based comparison for most cases
    - LLM verification for edge cases (if enabled)
    """
    
    def __init__(self, threshold: float = 0.75, use_hybrid: bool = None):
        """
        Args:
            threshold: Cosine similarity threshold (0.75 = more lenient for catching similar questions)
            use_hybrid: Use hybrid LLM approach for edge cases (default from settings)
        """
        self.threshold = threshold
        
        # Use hybrid approach if enabled in settings
        if use_hybrid is None:
            use_hybrid = settings.SIMILARITY_DETECTION.get('USE_HYBRID_APPROACH', False)
        self.use_hybrid = use_hybrid
        
        # Lazy load hybrid LLM service
        self._hybrid_llm = None
    
    def _get_hybrid_llm(self):
        """Lazy load hybrid LLM service."""
        if self._hybrid_llm is None and self.use_hybrid:
            try:
                from apps.analysis.services.hybrid_llm_service import HybridLLMService
                self._hybrid_llm = HybridLLMService()
                logger.info("✓ Hybrid LLM service initialized for similarity detection")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid LLM service: {e}")
        return self._hybrid_llm
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0 to 1)
        """
        # Reshape to 2D if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all embeddings.
        
        Args:
            embeddings: 2D array of embeddings [n_samples, embedding_dim]
            
        Returns:
            Similarity matrix [n_samples, n_samples]
        """
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def find_similar_pairs(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float, str, str]]:
        """
        Find all pairs of similar questions above threshold.
        
        Args:
            embeddings: Question embeddings
            texts: Corresponding question texts
            threshold: Override default threshold
            
        Returns:
            List of (idx1, idx2, similarity, text1, text2) tuples
        """
        threshold = threshold or self.threshold
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(embeddings)
        
        # Find pairs above threshold (excluding diagonal)
        n = len(embeddings)
        pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only
                similarity = sim_matrix[i][j]
                if similarity >= threshold:
                    pairs.append((
                        i,
                        j,
                        similarity,
                        texts[i],
                        texts[j]
                    ))
        
        # Sort by similarity (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(pairs)} similar pairs (threshold={threshold})")
        return pairs
    
    def is_duplicate(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
        marks1: Optional[int] = None,
        marks2: Optional[int] = None
    ) -> Tuple[bool, float, str, str]:
        """
        Check if two questions are duplicates (same meaning, reworded).
        Uses hybrid approach if enabled and texts are provided.
        
        Args:
            embedding1: First question embedding
            embedding2: Second question embedding
            text1: Optional first question text (for hybrid LLM)
            text2: Optional second question text (for hybrid LLM)
            marks1: Optional marks for first question
            marks2: Optional marks for second question
        
        Returns:
            Tuple of (is_similar, similarity_score, method, reason)
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        
        # If hybrid approach is enabled and we have text
        if self.use_hybrid and text1 and text2:
            hybrid_llm = self._get_hybrid_llm()
            if hybrid_llm:
                try:
                    is_similar, conf, method, reason = hybrid_llm.are_questions_similar(
                        text1, text2, marks1, marks2
                    )
                    return is_similar, conf, method, reason
                except Exception as e:
                    logger.warning(f"Hybrid LLM check failed: {e}, falling back to embedding")
        
        # Fallback to simple threshold check
        is_similar = similarity >= self.threshold
        return is_similar, similarity, 'embedding', f'Embedding similarity: {similarity:.3f}'
    
    def group_similar_questions(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[dict]] = None
    ) -> List[List[int]]:
        """
        Group questions into clusters of similar questions.
        Each cluster represents the same topic/question asked multiple times.
        
        Args:
            embeddings: Question embeddings
            texts: Question texts
            metadata: Optional metadata for each question
            
        Returns:
            List of clusters, where each cluster is a list of question indices
        """
        # Use agglomerative clustering with distance threshold
        distance_threshold = 1 - self.threshold  # Convert similarity to distance
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster label
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Convert to list and sort by cluster size (most repeated first)
        cluster_list = sorted(clusters.values(), key=len, reverse=True)
        
        logger.info(f"Grouped {len(embeddings)} questions into {len(cluster_list)} clusters")
        return cluster_list
