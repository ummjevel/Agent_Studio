"""
Pattern learning system that automatically extracts and improves patterns
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics

from .pattern_store import WorkflowPatternStore, WorkflowPattern
from .pattern_matcher import PatternMatcher
from ..workflow.graph import WorkflowGraph


class PatternLearner:
    """
    Learns workflow patterns from successful executions and improves them over time.
    
    This implements the self-evolution aspect for workflow patterns,
    automatically identifying successful patterns and refining them.
    """
    
    def __init__(self, pattern_store: WorkflowPatternStore):
        """
        Initialize the pattern learner.
        
        Args:
            pattern_store: Store for managing patterns
        """
        self.pattern_store = pattern_store
        self.pattern_matcher = PatternMatcher()
        
        # Learning configuration
        self.min_success_threshold = 3  # Minimum successes to create a pattern
        self.pattern_merge_threshold = 0.85  # Similarity threshold for merging patterns
        self.improvement_threshold = 0.1  # Minimum improvement to update a pattern
    
    def learn_from_workflow(
        self,
        workflow: WorkflowGraph,
        execution_result: Dict[str, Any],
        task_description: str,
        task_type: str,
        domain: Optional[str] = None
    ) -> Optional[str]:
        """
        Learn from a workflow execution.
        
        Args:
            workflow: The executed workflow
            execution_result: Results from execution (success, performance metrics, etc.)
            task_description: Description of the task that was solved
            task_type: Type of task
            domain: Application domain
            
        Returns:
            Pattern ID if a pattern was created or updated, None otherwise
        """
        success = execution_result.get("success", False)
        performance_score = execution_result.get("performance_score", 0.0)
        execution_time_ms = execution_result.get("execution_time_ms", 0.0)
        cost_usd = execution_result.get("cost_usd", 0.0)
        
        if not success:
            # Don't learn from failed executions
            return None
        
        # Extract keywords from task description
        keywords = self.pattern_matcher._extract_keywords(task_description)
        
        # Check if this workflow is similar to existing patterns
        existing_patterns = self.pattern_store.find_patterns(
            task_type=task_type,
            keywords=list(keywords),
            domain=domain,
            min_confidence=0.0  # Consider all patterns for learning
        )
        
        # Find the most similar existing pattern
        best_match = None
        best_similarity = 0.0
        
        if existing_patterns:
            matches = self.pattern_matcher.match_workflow_structure(workflow, existing_patterns)
            if matches:
                best_match, best_similarity = matches[0]
        
        if best_match and best_similarity >= self.pattern_merge_threshold:
            # Update existing pattern
            return self._update_existing_pattern(
                best_match,
                workflow,
                execution_result,
                task_description,
                keywords,
                domain
            )
        else:
            # Create new pattern
            return self._create_new_pattern(
                workflow,
                execution_result,
                task_description,
                task_type,
                keywords,
                domain
            )
    
    def _update_existing_pattern(
        self,
        pattern: WorkflowPattern,
        workflow: WorkflowGraph,
        execution_result: Dict[str, Any],
        task_description: str,
        keywords: set,
        domain: Optional[str]
    ) -> str:
        """Update an existing pattern with new execution data"""
        
        # Update performance metrics
        self.pattern_store.update_pattern_performance(
            pattern_id=pattern.id,
            success=True,
            performance_score=execution_result.get("performance_score"),
            execution_time_ms=execution_result.get("execution_time_ms"),
            cost_usd=execution_result.get("cost_usd")
        )
        
        # Update pattern metadata
        updated_pattern = self.pattern_store.patterns[pattern.id]
        
        # Add new keywords
        updated_pattern.keywords.update(keywords)
        
        # Add domain if provided
        if domain:
            updated_pattern.domains.add(domain)
        
        return pattern.id
    
    def _create_new_pattern(
        self,
        workflow: WorkflowGraph,
        execution_result: Dict[str, Any],
        task_description: str,
        task_type: str,
        keywords: set,
        domain: Optional[str]
    ) -> str:
        """Create a new pattern from a successful workflow"""
        
        # Extract pattern from workflow
        pattern = self.pattern_store.extract_pattern_from_workflow(
            workflow=workflow,
            task_type=task_type,
            keywords=list(keywords),
            domain=domain
        )
        
        # Initialize with first execution data
        pattern.success_count = 1
        pattern.avg_performance_score = execution_result.get("performance_score", 0.0)
        pattern.avg_execution_time_ms = execution_result.get("execution_time_ms", 0.0)
        pattern.avg_cost_usd = execution_result.get("cost_usd", 0.0)
        
        # Add to store
        return self.pattern_store.add_pattern(pattern)
    
    def analyze_pattern_evolution(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze how patterns have evolved over time.
        
        Args:
            days_back: Number of days to look back for analysis
            
        Returns:
            Analysis results
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Analyze patterns created or updated since cutoff
        recent_patterns = []
        for pattern in self.pattern_store.patterns.values():
            if pattern.created_at >= cutoff_date or (pattern.last_used and pattern.last_used >= cutoff_date):
                recent_patterns.append(pattern)
        
        if not recent_patterns:
            return {"message": "No recent pattern activity"}
        
        # Calculate statistics
        success_rates = [p.success_rate for p in recent_patterns]
        confidence_scores = [p.confidence_score for p in recent_patterns]
        performance_scores = [p.avg_performance_score for p in recent_patterns if p.avg_performance_score > 0]
        
        analysis = {
            "period_days": days_back,
            "patterns_analyzed": len(recent_patterns),
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
            "avg_confidence_score": statistics.mean(confidence_scores) if confidence_scores else 0,
            "avg_performance_score": statistics.mean(performance_scores) if performance_scores else 0,
            "total_pattern_uses": sum(p.success_count + p.failure_count for p in recent_patterns),
            "most_used_pattern": None,
            "highest_performing_pattern": None
        }
        
        # Find most used pattern
        if recent_patterns:
            most_used = max(recent_patterns, key=lambda p: p.success_count + p.failure_count)
            analysis["most_used_pattern"] = {
                "id": most_used.id,
                "name": most_used.name,
                "total_uses": most_used.success_count + most_used.failure_count
            }
        
        # Find highest performing pattern
        if recent_patterns:
            highest_performing = max(recent_patterns, key=lambda p: p.confidence_score)
            analysis["highest_performing_pattern"] = {
                "id": highest_performing.id,
                "name": highest_performing.name,
                "confidence_score": highest_performing.confidence_score
            }
        
        return analysis
    
    def identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify patterns that could be improved or merged.
        
        Returns:
            List of improvement opportunities
        """
        opportunities = []
        patterns = list(self.pattern_store.patterns.values())
        
        # Find patterns with low performance that could be improved
        for pattern in patterns:
            total_uses = pattern.success_count + pattern.failure_count
            
            if total_uses >= 5 and pattern.confidence_score < 0.6:
                opportunities.append({
                    "type": "low_performance",
                    "pattern_id": pattern.id,
                    "pattern_name": pattern.name,
                    "confidence_score": pattern.confidence_score,
                    "total_uses": total_uses,
                    "recommendation": "Consider removing or refining this pattern"
                })
        
        # Find similar patterns that could be merged
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                
                if similarity >= 0.75:  # High similarity threshold
                    opportunities.append({
                        "type": "merge_candidate",
                        "pattern1_id": pattern1.id,
                        "pattern1_name": pattern1.name,
                        "pattern2_id": pattern2.id,
                        "pattern2_name": pattern2.name,
                        "similarity": similarity,
                        "recommendation": "Consider merging these similar patterns"
                    })
        
        return opportunities
    
    def _calculate_pattern_similarity(self, pattern1: WorkflowPattern, pattern2: WorkflowPattern) -> float:
        """Calculate similarity between two patterns"""
        
        # Structural similarity
        from difflib import SequenceMatcher
        structural_sim = SequenceMatcher(None, pattern1.node_pattern, pattern2.node_pattern).ratio()
        
        # Contextual similarity
        common_task_types = pattern1.task_types & pattern2.task_types
        all_task_types = pattern1.task_types | pattern2.task_types
        task_type_sim = len(common_task_types) / max(len(all_task_types), 1)
        
        common_keywords = pattern1.keywords & pattern2.keywords
        all_keywords = pattern1.keywords | pattern2.keywords
        keyword_sim = len(common_keywords) / max(len(all_keywords), 1)
        
        # Complexity similarity
        complexity_sim = 1.0 if pattern1.complexity_level == pattern2.complexity_level else 0.5
        
        # Combine similarities
        total_similarity = (
            structural_sim * 0.4 +
            task_type_sim * 0.3 +
            keyword_sim * 0.2 +
            complexity_sim * 0.1
        )
        
        return total_similarity
    
    def suggest_pattern_optimizations(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Suggest optimizations for a specific pattern.
        
        Args:
            pattern_id: ID of pattern to analyze
            
        Returns:
            List of optimization suggestions
        """
        if pattern_id not in self.pattern_store.patterns:
            return []
        
        pattern = self.pattern_store.patterns[pattern_id]
        suggestions = []
        
        # Analyze performance
        if pattern.avg_performance_score < 5.0:  # Assuming 0-10 scale
            suggestions.append({
                "type": "performance",
                "issue": "Low average performance score",
                "current_score": pattern.avg_performance_score,
                "suggestion": "Review and optimize the workflow steps in this pattern"
            })
        
        # Analyze cost efficiency
        total_uses = pattern.success_count + pattern.failure_count
        if total_uses > 5 and pattern.avg_cost_usd > 1.0:  # High cost threshold
            suggestions.append({
                "type": "cost",
                "issue": "High average cost",
                "current_cost": pattern.avg_cost_usd,
                "suggestion": "Consider using more cost-efficient models or optimizing the workflow"
            })
        
        # Analyze execution time
        if pattern.avg_execution_time_ms > 30000:  # More than 30 seconds
            suggestions.append({
                "type": "latency",
                "issue": "High execution time",
                "current_time_ms": pattern.avg_execution_time_ms,
                "suggestion": "Optimize workflow for faster execution or consider parallel processing"
            })
        
        # Analyze usage patterns
        if pattern.success_rate < 80 and total_uses >= 5:
            suggestions.append({
                "type": "reliability",
                "issue": "Low success rate",
                "current_rate": pattern.success_rate,
                "suggestion": "Review failure cases and add error handling or validation steps"
            })
        
        return suggestions
    
    def auto_optimize_patterns(self, min_uses: int = 10) -> Dict[str, Any]:
        """
        Automatically optimize patterns based on performance data.
        
        Args:
            min_uses: Minimum number of uses before optimizing
            
        Returns:
            Summary of optimizations performed
        """
        optimizations = {
            "patterns_optimized": 0,
            "patterns_removed": 0,
            "patterns_merged": 0,
            "optimizations_applied": []
        }
        
        # Remove low-performing patterns
        removed_count = self.pattern_store.cleanup_low_performing_patterns(
            min_confidence=0.3,
            min_uses=min_uses
        )
        optimizations["patterns_removed"] = removed_count
        
        # Identify merge opportunities and implement them
        opportunities = self.identify_improvement_opportunities()
        
        merge_candidates = [opp for opp in opportunities if opp["type"] == "merge_candidate"]
        
        # Simple merging strategy: merge patterns with very high similarity
        for candidate in merge_candidates:
            if candidate["similarity"] >= 0.9:
                # Merge patterns (simplified - in practice, this would be more sophisticated)
                pattern1_id = candidate["pattern1_id"]
                pattern2_id = candidate["pattern2_id"]
                
                if pattern1_id in self.pattern_store.patterns and pattern2_id in self.pattern_store.patterns:
                    # Keep the pattern with better performance
                    pattern1 = self.pattern_store.patterns[pattern1_id]
                    pattern2 = self.pattern_store.patterns[pattern2_id]
                    
                    if pattern1.confidence_score >= pattern2.confidence_score:
                        # Merge pattern2 into pattern1
                        pattern1.success_count += pattern2.success_count
                        pattern1.failure_count += pattern2.failure_count
                        pattern1.keywords.update(pattern2.keywords)
                        pattern1.task_types.update(pattern2.task_types)
                        pattern1.domains.update(pattern2.domains)
                        
                        # Remove pattern2
                        self.pattern_store.remove_pattern(pattern2_id)
                        optimizations["patterns_merged"] += 1
                        optimizations["optimizations_applied"].append(f"Merged pattern {pattern2_id} into {pattern1_id}")
        
        return optimizations