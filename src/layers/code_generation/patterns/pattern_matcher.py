"""
Pattern matching algorithms for finding similar workflow structures
"""

from typing import List, Dict, Any, Tuple, Optional
import re
from difflib import SequenceMatcher

from .pattern_store import WorkflowPattern
from ..workflow.graph import WorkflowGraph


class PatternMatcher:
    """
    Matches new tasks against existing workflow patterns.
    
    This uses various similarity metrics to find the most relevant
    patterns for a given task description and requirements.
    """
    
    def __init__(self):
        self.similarity_weights = {
            "structural": 0.3,      # Workflow structure similarity
            "semantic": 0.4,        # Task description similarity
            "contextual": 0.2,      # Context/domain similarity
            "performance": 0.1      # Historical performance
        }
    
    def find_best_matches(
        self,
        task_description: str,
        patterns: List[WorkflowPattern],
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity_level: Optional[str] = None,
        max_matches: int = 5
    ) -> List[Tuple[WorkflowPattern, float]]:
        """
        Find the best matching patterns for a given task.
        
        Args:
            task_description: Description of the task
            patterns: List of patterns to match against
            task_type: Type of task (if known)
            domain: Application domain (if known)
            complexity_level: Required complexity level
            max_matches: Maximum number of matches to return
            
        Returns:
            List of (pattern, similarity_score) tuples sorted by similarity
        """
        matches = []
        
        for pattern in patterns:
            similarity_score = self.calculate_similarity(
                task_description=task_description,
                pattern=pattern,
                task_type=task_type,
                domain=domain,
                complexity_level=complexity_level
            )
            
            matches.append((pattern, similarity_score))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:max_matches]
    
    def calculate_similarity(
        self,
        task_description: str,
        pattern: WorkflowPattern,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity_level: Optional[str] = None
    ) -> float:
        """
        Calculate similarity score between a task and a pattern.
        
        Args:
            task_description: Description of the task
            pattern: Pattern to match against
            task_type: Type of task (if known)
            domain: Application domain (if known)
            complexity_level: Required complexity level
            
        Returns:
            Similarity score between 0 and 1
        """
        # Calculate individual similarity components
        semantic_sim = self._calculate_semantic_similarity(task_description, pattern)
        contextual_sim = self._calculate_contextual_similarity(task_type, domain, pattern)
        performance_sim = self._calculate_performance_similarity(pattern)
        complexity_sim = self._calculate_complexity_similarity(complexity_level, pattern)
        
        # Combine similarities using weights
        total_similarity = (
            semantic_sim * self.similarity_weights["semantic"] +
            contextual_sim * self.similarity_weights["contextual"] +
            performance_sim * self.similarity_weights["performance"] +
            complexity_sim * self.similarity_weights["structural"]  # Using structural weight for complexity
        )
        
        return min(total_similarity, 1.0)
    
    def _calculate_semantic_similarity(self, task_description: str, pattern: WorkflowPattern) -> float:
        """Calculate semantic similarity based on text analysis"""
        
        # Extract keywords from task description
        task_keywords = self._extract_keywords(task_description)
        
        # Calculate keyword overlap
        if not pattern.keywords:
            keyword_overlap = 0.0
        else:
            common_keywords = task_keywords & pattern.keywords
            keyword_overlap = len(common_keywords) / max(len(task_keywords), len(pattern.keywords))
        
        # Calculate text similarity with pattern description
        text_similarity = SequenceMatcher(None, task_description.lower(), pattern.description.lower()).ratio()
        
        # Combine keyword and text similarities
        semantic_similarity = (keyword_overlap * 0.6) + (text_similarity * 0.4)
        
        return semantic_similarity
    
    def _calculate_contextual_similarity(
        self, 
        task_type: Optional[str], 
        domain: Optional[str], 
        pattern: WorkflowPattern
    ) -> float:
        """Calculate contextual similarity based on task type and domain"""
        
        contextual_score = 0.0
        
        # Task type similarity
        if task_type and pattern.task_types:
            if task_type in pattern.task_types:
                contextual_score += 0.7
            else:
                # Check for related task types
                related_score = self._calculate_task_type_relatedness(task_type, pattern.task_types)
                contextual_score += related_score * 0.3
        
        # Domain similarity
        if domain and pattern.domains:
            if domain in pattern.domains:
                contextual_score += 0.3
            else:
                # Check for related domains
                related_score = self._calculate_domain_relatedness(domain, pattern.domains)
                contextual_score += related_score * 0.1
        
        # If no context provided, use neutral score
        if not task_type and not domain:
            contextual_score = 0.5
        
        return min(contextual_score, 1.0)
    
    def _calculate_performance_similarity(self, pattern: WorkflowPattern) -> float:
        """Calculate similarity based on pattern performance history"""
        
        # Use confidence score as performance indicator
        return pattern.confidence_score
    
    def _calculate_complexity_similarity(
        self, 
        required_complexity: Optional[str], 
        pattern: WorkflowPattern
    ) -> float:
        """Calculate similarity based on complexity level match"""
        
        if not required_complexity:
            return 0.5  # Neutral score if no requirement
        
        complexity_map = {"simple": 1, "medium": 2, "complex": 3}
        
        required_level = complexity_map.get(required_complexity, 2)
        pattern_level = complexity_map.get(pattern.complexity_level, 2)
        
        # Calculate distance-based similarity
        max_distance = 2  # Maximum possible distance
        distance = abs(required_level - pattern_level)
        similarity = 1 - (distance / max_distance)
        
        return similarity
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        
        # Simple keyword extraction - could be enhanced with NLP
        text = text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (alphabetic characters only)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter out stop words and short words
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        
        return keywords
    
    def _calculate_task_type_relatedness(self, task_type: str, pattern_task_types: set) -> float:
        """Calculate relatedness between task types"""
        
        # Define task type relationships
        task_relationships = {
            "qa": {"simple_qa", "question_answering", "information_retrieval"},
            "analysis": {"research_analysis", "data_analysis", "evaluation"},
            "generation": {"code_generation", "content_generation", "creation"},
            "processing": {"data_processing", "transformation", "computation"},
            "reasoning": {"multi_step_reasoning", "logic", "problem_solving"}
        }
        
        # Find related task types
        related_types = set()
        for category, types in task_relationships.items():
            if task_type in types:
                related_types.update(types)
        
        # Calculate overlap with pattern task types
        if not related_types:
            return 0.0
        
        overlap = related_types & pattern_task_types
        return len(overlap) / len(related_types)
    
    def _calculate_domain_relatedness(self, domain: str, pattern_domains: set) -> float:
        """Calculate relatedness between domains"""
        
        # Define domain relationships
        domain_relationships = {
            "business": {"finance", "marketing", "sales", "operations"},
            "technology": {"software", "ai", "data", "web", "mobile"},
            "research": {"academic", "scientific", "analysis", "investigation"},
            "creative": {"design", "writing", "content", "media"}
        }
        
        # Find related domains
        related_domains = set()
        for category, domains in domain_relationships.items():
            if domain in domains:
                related_domains.update(domains)
        
        # Calculate overlap with pattern domains
        if not related_domains:
            return 0.0
        
        overlap = related_domains & pattern_domains
        return len(overlap) / len(related_domains)
    
    def match_workflow_structure(self, workflow: WorkflowGraph, patterns: List[WorkflowPattern]) -> List[Tuple[WorkflowPattern, float]]:
        """
        Match a workflow structure against existing patterns.
        
        Args:
            workflow: Workflow to match
            patterns: Patterns to match against
            
        Returns:
            List of (pattern, structural_similarity) tuples
        """
        workflow_structure = self._extract_workflow_structure(workflow)
        matches = []
        
        for pattern in patterns:
            structural_similarity = self._calculate_structural_similarity(
                workflow_structure, 
                pattern
            )
            matches.append((pattern, structural_similarity))
        
        # Sort by structural similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def _extract_workflow_structure(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """Extract structural features from a workflow"""
        
        # Node type sequence
        node_types = [node.node_type.value for node in workflow.nodes.values()]
        
        # Edge patterns
        edge_patterns = []
        for edge in workflow.edges:
            edge_patterns.append({
                "from_type": workflow.nodes[edge.from_node].node_type.value,
                "to_type": workflow.nodes[edge.to_node].node_type.value
            })
        
        return {
            "node_types": node_types,
            "edge_patterns": edge_patterns,
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges)
        }
    
    def _calculate_structural_similarity(
        self, 
        workflow_structure: Dict[str, Any], 
        pattern: WorkflowPattern
    ) -> float:
        """Calculate structural similarity between workflow and pattern"""
        
        # Node type sequence similarity
        node_similarity = SequenceMatcher(
            None, 
            workflow_structure["node_types"], 
            pattern.node_pattern
        ).ratio()
        
        # Size similarity (penalize large differences)
        size_diff = abs(workflow_structure["node_count"] - len(pattern.node_pattern))
        max_size = max(workflow_structure["node_count"], len(pattern.node_pattern))
        size_similarity = 1 - (size_diff / max(max_size, 1))
        
        # Combine similarities
        structural_similarity = (node_similarity * 0.7) + (size_similarity * 0.3)
        
        return structural_similarity