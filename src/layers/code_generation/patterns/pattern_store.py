"""
Storage and management of successful workflow patterns
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

from ..workflow.graph import WorkflowGraph


class WorkflowPattern(BaseModel):
    """
    Represents a learned workflow pattern that can be reused.
    
    This captures successful workflow structures along with their
    performance characteristics and usage context.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Pattern structure
    node_pattern: List[str] = Field(description="Sequence of node types")
    edge_pattern: List[Dict[str, str]] = Field(description="Edge connection patterns")
    complexity_level: str = Field(description="simple, medium, complex")
    
    # Success metrics
    success_count: int = Field(default=0, description="Number of successful uses")
    failure_count: int = Field(default=0, description="Number of failed uses")
    avg_performance_score: float = Field(default=0.0, description="Average performance rating")
    avg_execution_time_ms: float = Field(default=0.0, description="Average execution time")
    avg_cost_usd: float = Field(default=0.0, description="Average cost")
    
    # Usage context
    task_types: Set[str] = Field(default_factory=set, description="Types of tasks this pattern works for")
    keywords: Set[str] = Field(default_factory=set, description="Keywords associated with successful uses")
    domains: Set[str] = Field(default_factory=set, description="Application domains")
    
    # Pattern metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    source_workflow_id: Optional[str] = None
    
    # Variations
    variations: List[str] = Field(default_factory=list, description="IDs of pattern variations")
    parent_pattern: Optional[str] = None
    
    class Config:
        use_enum_values = True
        json_encoders = {
            set: list,  # Convert sets to lists for JSON serialization
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage"""
        total_uses = self.success_count + self.failure_count
        if total_uses == 0:
            return 0.0
        return (self.success_count / total_uses) * 100
    
    @property
    def confidence_score(self) -> float:
        """
        Calculate confidence score based on usage and performance.
        
        Higher values indicate more reliable patterns.
        """
        # Base score from success rate
        base_score = self.success_rate / 100
        
        # Usage factor (more uses = higher confidence)
        usage_factor = min(self.success_count / 10, 1.0)  # Cap at 10 uses
        
        # Performance factor
        performance_factor = min(self.avg_performance_score / 10, 1.0)  # Assuming 0-10 scale
        
        # Combine factors
        confidence = (base_score * 0.5) + (usage_factor * 0.3) + (performance_factor * 0.2)
        
        return min(confidence, 1.0)


class WorkflowPatternStore:
    """
    Storage and management system for workflow patterns.
    
    This system learns from successful workflows and stores patterns
    that can be reused for similar tasks in the future.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the pattern store.
        
        Args:
            storage_path: Path to store patterns (if None, uses in-memory storage)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.patterns: Dict[str, WorkflowPattern] = {}
        self.pattern_index: Dict[str, Set[str]] = {
            "task_types": {},
            "keywords": {},
            "domains": {},
            "complexity": {}
        }
        
        # Load existing patterns if storage path is provided
        if self.storage_path and self.storage_path.exists():
            self._load_patterns()
    
    def add_pattern(self, pattern: WorkflowPattern) -> str:
        """
        Add a new pattern to the store.
        
        Args:
            pattern: Pattern to add
            
        Returns:
            Pattern ID
        """
        # Store the pattern
        self.patterns[pattern.id] = pattern
        
        # Update indices
        self._update_indices(pattern)
        
        # Persist if storage is configured
        if self.storage_path:
            self._save_patterns()
        
        return pattern.id
    
    def update_pattern_performance(
        self,
        pattern_id: str,
        success: bool,
        performance_score: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        cost_usd: Optional[float] = None
    ) -> bool:
        """
        Update pattern performance metrics.
        
        Args:
            pattern_id: ID of the pattern to update
            success: Whether the pattern use was successful
            performance_score: Performance rating (0-10 scale)
            execution_time_ms: Execution time in milliseconds
            cost_usd: Cost in USD
            
        Returns:
            True if pattern was updated, False if not found
        """
        if pattern_id not in self.patterns:
            return False
        
        pattern = self.patterns[pattern_id]
        
        # Update counts
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1
        
        pattern.last_used = datetime.now()
        
        # Update averages if values provided
        if performance_score is not None:
            # Running average calculation
            total_uses = pattern.success_count + pattern.failure_count
            pattern.avg_performance_score = (
                (pattern.avg_performance_score * (total_uses - 1) + performance_score) / total_uses
            )
        
        if execution_time_ms is not None:
            total_uses = pattern.success_count + pattern.failure_count
            pattern.avg_execution_time_ms = (
                (pattern.avg_execution_time_ms * (total_uses - 1) + execution_time_ms) / total_uses
            )
        
        if cost_usd is not None:
            total_uses = pattern.success_count + pattern.failure_count
            pattern.avg_cost_usd = (
                (pattern.avg_cost_usd * (total_uses - 1) + cost_usd) / total_uses
            )
        
        # Persist changes
        if self.storage_path:
            self._save_patterns()
        
        return True
    
    def find_patterns(
        self,
        task_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        domain: Optional[str] = None,
        complexity_level: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[WorkflowPattern]:
        """
        Find patterns matching the given criteria.
        
        Args:
            task_type: Type of task to find patterns for
            keywords: Keywords to match against
            domain: Application domain
            complexity_level: Required complexity level
            min_confidence: Minimum confidence score
            limit: Maximum number of patterns to return
            
        Returns:
            List of matching patterns sorted by confidence
        """
        candidates = set(self.patterns.keys())
        
        # Filter by task type
        if task_type and task_type in self.pattern_index["task_types"]:
            candidates &= self.pattern_index["task_types"][task_type]
        
        # Filter by keywords
        if keywords:
            keyword_matches = set()
            for keyword in keywords:
                if keyword in self.pattern_index["keywords"]:
                    keyword_matches |= self.pattern_index["keywords"][keyword]
            if keyword_matches:
                candidates &= keyword_matches
        
        # Filter by domain
        if domain and domain in self.pattern_index["domains"]:
            candidates &= self.pattern_index["domains"][domain]
        
        # Filter by complexity
        if complexity_level and complexity_level in self.pattern_index["complexity"]:
            candidates &= self.pattern_index["complexity"][complexity_level]
        
        # Get patterns and filter by confidence
        matching_patterns = []
        for pattern_id in candidates:
            pattern = self.patterns[pattern_id]
            if pattern.confidence_score >= min_confidence:
                matching_patterns.append(pattern)
        
        # Sort by confidence score (descending)
        matching_patterns.sort(key=lambda p: p.confidence_score, reverse=True)
        
        return matching_patterns[:limit]
    
    def extract_pattern_from_workflow(
        self,
        workflow: WorkflowGraph,
        task_type: str,
        keywords: Optional[List[str]] = None,
        domain: Optional[str] = None
    ) -> WorkflowPattern:
        """
        Extract a reusable pattern from a successful workflow.
        
        Args:
            workflow: Workflow to extract pattern from
            task_type: Type of task this workflow solved
            keywords: Keywords associated with the task
            domain: Application domain
            
        Returns:
            Extracted workflow pattern
        """
        # Extract structural pattern
        node_pattern = [node.node_type.value for node in workflow.nodes.values()]
        
        edge_pattern = []
        for edge in workflow.edges:
            edge_pattern.append({
                "from_type": workflow.nodes[edge.from_node].node_type.value,
                "to_type": workflow.nodes[edge.to_node].node_type.value,
                "data_key": edge.data_key
            })
        
        # Determine complexity level
        stats = workflow.get_statistics()
        if stats["total_nodes"] <= 3:
            complexity_level = "simple"
        elif stats["total_nodes"] >= 7:
            complexity_level = "complex"
        else:
            complexity_level = "medium"
        
        # Create pattern
        pattern = WorkflowPattern(
            name=f"Pattern for {task_type}",
            description=f"Extracted from workflow: {workflow.name}",
            node_pattern=node_pattern,
            edge_pattern=edge_pattern,
            complexity_level=complexity_level,
            task_types={task_type},
            keywords=set(keywords or []),
            domains={domain} if domain else set(),
            source_workflow_id=workflow.id
        )
        
        return pattern
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns"""
        if not self.patterns:
            return {"total_patterns": 0}
        
        total_patterns = len(self.patterns)
        avg_confidence = sum(p.confidence_score for p in self.patterns.values()) / total_patterns
        
        complexity_dist = {}
        task_type_dist = {}
        
        for pattern in self.patterns.values():
            # Complexity distribution
            complexity_dist[pattern.complexity_level] = complexity_dist.get(pattern.complexity_level, 0) + 1
            
            # Task type distribution
            for task_type in pattern.task_types:
                task_type_dist[task_type] = task_type_dist.get(task_type, 0) + 1
        
        return {
            "total_patterns": total_patterns,
            "avg_confidence_score": avg_confidence,
            "complexity_distribution": complexity_dist,
            "task_type_distribution": task_type_dist,
            "total_successful_uses": sum(p.success_count for p in self.patterns.values()),
            "total_failed_uses": sum(p.failure_count for p in self.patterns.values())
        }
    
    def cleanup_low_performing_patterns(self, min_confidence: float = 0.3, min_uses: int = 5):
        """
        Remove patterns with low performance.
        
        Args:
            min_confidence: Minimum confidence score to keep
            min_uses: Minimum number of uses before considering for removal
        """
        patterns_to_remove = []
        
        for pattern_id, pattern in self.patterns.items():
            total_uses = pattern.success_count + pattern.failure_count
            if total_uses >= min_uses and pattern.confidence_score < min_confidence:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            self.remove_pattern(pattern_id)
        
        return len(patterns_to_remove)
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the store.
        
        Args:
            pattern_id: ID of pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        if pattern_id not in self.patterns:
            return False
        
        # Remove from indices
        pattern = self.patterns[pattern_id]
        self._remove_from_indices(pattern)
        
        # Remove pattern
        del self.patterns[pattern_id]
        
        # Persist changes
        if self.storage_path:
            self._save_patterns()
        
        return True
    
    def _update_indices(self, pattern: WorkflowPattern):
        """Update search indices with pattern data"""
        
        # Task types index
        for task_type in pattern.task_types:
            if task_type not in self.pattern_index["task_types"]:
                self.pattern_index["task_types"][task_type] = set()
            self.pattern_index["task_types"][task_type].add(pattern.id)
        
        # Keywords index
        for keyword in pattern.keywords:
            if keyword not in self.pattern_index["keywords"]:
                self.pattern_index["keywords"][keyword] = set()
            self.pattern_index["keywords"][keyword].add(pattern.id)
        
        # Domains index
        for domain in pattern.domains:
            if domain not in self.pattern_index["domains"]:
                self.pattern_index["domains"][domain] = set()
            self.pattern_index["domains"][domain].add(pattern.id)
        
        # Complexity index
        complexity = pattern.complexity_level
        if complexity not in self.pattern_index["complexity"]:
            self.pattern_index["complexity"][complexity] = set()
        self.pattern_index["complexity"][complexity].add(pattern.id)
    
    def _remove_from_indices(self, pattern: WorkflowPattern):
        """Remove pattern from search indices"""
        
        # Remove from all relevant indices
        for task_type in pattern.task_types:
            if task_type in self.pattern_index["task_types"]:
                self.pattern_index["task_types"][task_type].discard(pattern.id)
        
        for keyword in pattern.keywords:
            if keyword in self.pattern_index["keywords"]:
                self.pattern_index["keywords"][keyword].discard(pattern.id)
        
        for domain in pattern.domains:
            if domain in self.pattern_index["domains"]:
                self.pattern_index["domains"][domain].discard(pattern.id)
        
        complexity = pattern.complexity_level
        if complexity in self.pattern_index["complexity"]:
            self.pattern_index["complexity"][complexity].discard(pattern.id)
    
    def _save_patterns(self):
        """Save patterns to persistent storage"""
        if not self.storage_path:
            return
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        patterns_data = {}
        for pattern_id, pattern in self.patterns.items():
            patterns_data[pattern_id] = pattern.model_dump()
        
        # Save to file
        with open(self.storage_path, 'w') as f:
            json.dump(patterns_data, f, indent=2, default=str)
    
    def _load_patterns(self):
        """Load patterns from persistent storage"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                patterns_data = json.load(f)
            
            # Reconstruct patterns
            for pattern_id, pattern_dict in patterns_data.items():
                # Convert lists back to sets where needed
                if 'task_types' in pattern_dict and isinstance(pattern_dict['task_types'], list):
                    pattern_dict['task_types'] = set(pattern_dict['task_types'])
                if 'keywords' in pattern_dict and isinstance(pattern_dict['keywords'], list):
                    pattern_dict['keywords'] = set(pattern_dict['keywords'])
                if 'domains' in pattern_dict and isinstance(pattern_dict['domains'], list):
                    pattern_dict['domains'] = set(pattern_dict['domains'])
                
                pattern = WorkflowPattern(**pattern_dict)
                self.patterns[pattern_id] = pattern
                self._update_indices(pattern)
        
        except Exception as e:
            print(f"Error loading patterns: {e}")
            # Continue with empty store if loading fails