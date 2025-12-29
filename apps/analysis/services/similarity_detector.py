"""
Similarity detection using cosine similarity.
Detects repeated/similar questions across years.
"""
import logging
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class SimilarityDetector:
    """
    Detects similar/repeated questions using cosine similarity.
    Implements the similarity threshold logic from master prompt.
    """
    
    def __init__(self, threshold: float = 0.75):
        """
        Args:
            threshold: Cosine similarity threshold (0.75 = more lenient for catching similar questions)
        """
        self.threshold = threshold
    
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
    
    def is_duplicate(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """
        Check if two questions are duplicates (same meaning, reworded).
        
        Returns:
            True if similarity >= threshold
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= self.threshold
    
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
