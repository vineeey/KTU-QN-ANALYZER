"""
Topic clustering and repetition analysis service.
Groups similar questions into topics and calculates priorities.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from django.db import transaction

from apps.questions.models import Question
from apps.subjects.models import Subject, Module
from apps.analytics.models import TopicCluster

logger = logging.getLogger(__name__)


class TopicClusteringService:
    """
    Service to cluster questions into topics and analyze repetition patterns.
    """
    
    def __init__(
        self,
        subject: Subject,
        similarity_threshold: float = 0.75,
        tier_1_threshold: int = 4,
        tier_2_threshold: int = 3,
        tier_3_threshold: int = 2
    ):
        self.subject = subject
        self.similarity_threshold = similarity_threshold
        self.tier_1_threshold = tier_1_threshold
        self.tier_2_threshold = tier_2_threshold
        self.tier_3_threshold = tier_3_threshold
    
    def analyze_subject(self) -> Dict[str, Any]:
        """
        Main entry point: analyze all questions for a subject and create topic clusters.
        
        Returns:
            Statistics about the clustering process
        """
        logger.info(f"Starting topic analysis for subject: {self.subject}")
        
        # Get all questions for this subject
        questions = Question.objects.filter(
            paper__subject=self.subject
        ).select_related('paper', 'module').order_by('module__number', 'question_number')
        
        if not questions.exists():
            logger.warning(f"No questions found for subject {self.subject}")
            return {'clusters_created': 0, 'questions_processed': 0}
        
        # Clear existing clusters for this subject
        with transaction.atomic():
            TopicCluster.objects.filter(subject=self.subject).delete()
        
        # Group questions by module
        modules = self.subject.modules.all()
        total_clusters = 0
        
        for module in modules:
            module_questions = questions.filter(module=module)
            if module_questions.exists():
                clusters_count = self._cluster_module_questions(module, module_questions)
                total_clusters += clusters_count
                logger.info(f"Created {clusters_count} clusters for {module}")
        
        # Handle unclassified questions
        unclassified = questions.filter(module__isnull=True)
        if unclassified.exists():
            clusters_count = self._cluster_module_questions(None, unclassified)
            total_clusters += clusters_count
            logger.info(f"Created {clusters_count} clusters for unclassified questions")
        
        return {
            'clusters_created': total_clusters,
            'questions_processed': questions.count()
        }
    
    def _cluster_module_questions(self, module: Optional[Module], questions) -> int:
        """
        Cluster questions within a single module.
        
        Returns:
            Number of clusters created
        """
        if not questions.exists():
            return 0
        
        # Group questions by similarity
        clusters = []
        processed = set()
        
        for q in questions:
            if q.id in processed:
                continue
            
            # Create a new cluster starting with this question
            cluster = {
                'representative': q,
                'questions': [q],
                'normalized_key': self._normalize_text(q.text),
            }
            processed.add(q.id)
            
            # Find similar questions
            for other_q in questions:
                if other_q.id in processed:
                    continue
                
                if self._are_similar(q, other_q):
                    cluster['questions'].append(other_q)
                    processed.add(other_q.id)
            
            clusters.append(cluster)
        
        # Create TopicCluster objects
        with transaction.atomic():
            for cluster_data in clusters:
                self._create_topic_cluster(module, cluster_data)
        
        return len(clusters)
    
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
        Extract a human-readable topic name from a question.
        """
        text = question.text
        
        # Remove question indicators
        text = re.sub(r'^(explain|define|describe|discuss|what|how|why|list|enumerate|state|elaborate|illustrate|classify|compare|differentiate)\s+', '', text, flags=re.IGNORECASE)
        
        # Take first part (up to 100 chars or first sentence)
        if '?' in text:
            text = text.split('?')[0]
        elif '.' in text:
            sentences = text.split('.')
            text = sentences[0] if len(sentences[0]) < 100 else sentences[0][:100]
        else:
            text = text[:100]
        
        # Capitalize first letter
        text = text.strip().capitalize()
        
        # If too long, truncate and add ellipsis
        if len(text) > 80:
            text = text[:77] + '...'
        
        return text
    
    def _create_topic_cluster(self, module: Optional[Module], cluster_data: Dict[str, Any]):
        """
        Create a TopicCluster object from cluster data.
        """
        representative = cluster_data['representative']
        questions = cluster_data['questions']
        
        # Extract topic name
        topic_name = self._extract_topic_name(representative)
        
        # Calculate statistics
        years = set()
        total_marks = 0
        part_a_count = 0
        part_b_count = 0
        
        for q in questions:
            # Get year from paper
            if q.paper.year:
                years.add(q.paper.year)
            
            # Add marks
            if q.marks:
                total_marks += q.marks
            
            # Count parts
            if q.part and q.part.upper() == 'A':
                part_a_count += 1
            elif q.part and q.part.upper() == 'B':
                part_b_count += 1
        
        frequency_count = len(years)
        years_appeared = sorted(list(years))
        
        # Create cluster
        with transaction.atomic():
            cluster = TopicCluster.objects.create(
                subject=self.subject,
                module=module,
                topic_name=topic_name,
                normalized_key=cluster_data['normalized_key'],
                representative_text=representative.text,
                frequency_count=frequency_count,
                years_appeared=years_appeared,
                total_marks=total_marks,
                part_a_count=part_a_count,
                part_b_count=part_b_count,
            )
            
            # Calculate and set priority tier
            cluster.calculate_priority_tier(
                self.tier_1_threshold,
                self.tier_2_threshold,
                self.tier_3_threshold
            )
            cluster.save()
            
            # Link questions to cluster
            for q in questions:
                q.topic_cluster = cluster
                q.save(update_fields=['topic_cluster'])
        
        logger.debug(f"Created cluster: {topic_name} ({frequency_count} occurrences)")


def analyze_subject_topics(
    subject: Subject,
    similarity_threshold: float = 0.75,
    tier_1_threshold: int = 4,
    tier_2_threshold: int = 3,
    tier_3_threshold: int = 2
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
