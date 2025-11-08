"""
Main WorkflowCodeGenerator class that integrates all generation approaches
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import uuid

from .workflow.graph import WorkflowGraph
from .hybrid_generator import HybridWorkflowGenerator, GenerationStrategy
from .templates.template_generator import TemplateBasedGenerator
from .llm_generators.llm_generator import LLMDirectGenerator
from .llm_client.client_factory import LLMClientFactory
from .patterns.pattern_store import WorkflowPatternStore
from .patterns.pattern_learner import PatternLearner
from .patterns.pattern_matcher import PatternMatcher
from .langgraph_integration.executor import LangGraphExecutor


class GenerationMode(str, Enum):
    """Available generation modes"""
    FAST = "fast"               # Quick generation using templates
    BALANCED = "balanced"       # Hybrid approach
    CREATIVE = "creative"       # LLM-heavy generation
    LEARNING = "learning"       # Pattern-based learning generation


class WorkflowCodeGenerator:
    """
    Main workflow code generator that integrates all generation approaches.
    
    This is the primary interface for Layer 3 (Code Generation) and provides
    a unified API for generating workflows using different strategies.
    
    Based on the design from introduce.md:
    - Workflow code synthesis
    - Node & edge construction  
    - Integration with other layers
    - Self-evolving capabilities
    """
    
    def __init__(
        self,
        llm_client_factory: Optional[LLMClientFactory] = None,
        pattern_store_path: Optional[str] = None,
        enable_learning: bool = True
    ):
        """
        Initialize the workflow code generator.
        
        Args:
            llm_client_factory: Factory for LLM clients (integrates with Layer 1)
            pattern_store_path: Path to store learned patterns
            enable_learning: Whether to enable pattern learning
        """
        # Initialize core components
        self.llm_client_factory = llm_client_factory or LLMClientFactory()
        
        # Initialize generators
        self.template_generator = TemplateBasedGenerator()
        self.llm_generator = LLMDirectGenerator(self.llm_client_factory)
        self.hybrid_generator = HybridWorkflowGenerator(self.llm_client_factory)
        
        # Initialize pattern learning components
        self.pattern_store = WorkflowPatternStore(pattern_store_path)
        self.pattern_learner = PatternLearner(self.pattern_store) if enable_learning else None
        self.pattern_matcher = PatternMatcher()
        
        # Initialize execution components
        self.executor = LangGraphExecutor()
        
        # Configuration
        self.default_mode = GenerationMode.BALANCED
        self.enable_learning = enable_learning
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "mode_usage": {mode.value: 0 for mode in GenerationMode},
            "avg_generation_time_ms": 0.0
        }
    
    def generate_workflow(
        self,
        task_description: str,
        mode: Optional[GenerationMode] = None,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity_hint: Optional[str] = None,
        budget_constraint: Optional[float] = None,
        latency_requirement: str = "normal",
        use_patterns: bool = True,
        **kwargs
    ) -> WorkflowGraph:
        """
        Generate a workflow for the given task.
        
        This is the main entry point that integrates with Layer 1 (Model Selection)
        and Layer 2 (Prompt Preprocessing) to generate optimal workflows.
        
        Args:
            task_description: Description of the task to create a workflow for
            mode: Generation mode to use
            task_type: Type of task (for pattern matching)
            domain: Application domain
            complexity_hint: Manual complexity hint
            budget_constraint: Maximum cost constraint
            latency_requirement: Latency requirement ("fast", "normal", "slow")
            use_patterns: Whether to use learned patterns
            **kwargs: Additional parameters
            
        Returns:
            Generated WorkflowGraph
        """
        start_time = datetime.now()
        
        # Update statistics
        self.generation_stats["total_generations"] += 1
        
        # Determine generation mode
        if mode is None:
            mode = self._auto_select_mode(
                task_description, complexity_hint, budget_constraint, latency_requirement
            )
        
        self.generation_stats["mode_usage"][mode.value] += 1
        
        try:
            # Check for existing patterns first if enabled
            workflow = None
            used_pattern = None
            
            if use_patterns and self.pattern_store:
                workflow, used_pattern = self._try_pattern_based_generation(
                    task_description, task_type, domain, complexity_hint
                )
            
            # If no suitable pattern found, use selected generation mode
            if workflow is None:
                workflow = self._generate_with_mode(
                    mode=mode,
                    task_description=task_description,
                    task_type=task_type,
                    domain=domain,
                    complexity_hint=complexity_hint,
                    budget_constraint=budget_constraint,
                    latency_requirement=latency_requirement,
                    **kwargs
                )
            
            # Add generation metadata
            generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            workflow.metadata.update({
                "generation_mode": mode.value,
                "generation_time_ms": generation_time_ms,
                "used_pattern": used_pattern.id if used_pattern else None,
                "layer3_generator": "WorkflowCodeGenerator",
                "budget_constraint": budget_constraint,
                "latency_requirement": latency_requirement
            })
            
            # Update statistics
            self.generation_stats["successful_generations"] += 1
            self._update_avg_generation_time(generation_time_ms)
            
            return workflow
            
        except Exception as e:
            # Update error statistics
            self.generation_stats["failed_generations"] += 1
            
            # Create fallback workflow
            fallback_workflow = self._create_fallback_workflow(task_description, str(e))
            
            generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_avg_generation_time(generation_time_ms)
            
            return fallback_workflow
    
    def execute_workflow(
        self,
        workflow: WorkflowGraph,
        initial_data: Optional[Dict[str, Any]] = None,
        learn_from_execution: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a generated workflow and optionally learn from the results.
        
        Args:
            workflow: Workflow to execute
            initial_data: Initial data for the workflow
            learn_from_execution: Whether to learn patterns from execution
            **kwargs: Additional execution parameters
            
        Returns:
            Execution results including performance metrics
        """
        # Execute the workflow
        execution_state = self.executor.execute_workflow(
            workflow=workflow,
            initial_data=initial_data,
            **kwargs
        )
        
        # Prepare execution results
        execution_summary = execution_state.get_execution_summary()
        
        execution_results = {
            "success": execution_summary["status"] == "completed",
            "execution_summary": execution_summary,
            "workflow_state": execution_state,
            "performance_metrics": self._calculate_performance_metrics(execution_state)
        }
        
        # Learn from execution if enabled
        if learn_from_execution and self.pattern_learner and execution_results["success"]:
            self._learn_from_execution(workflow, execution_results, **kwargs)
        
        return execution_results
    
    def optimize_workflow(
        self,
        workflow: WorkflowGraph,
        optimization_goals: Optional[List[str]] = None,
        performance_issues: Optional[List[str]] = None
    ) -> WorkflowGraph:
        """
        Optimize an existing workflow.
        
        Args:
            workflow: Workflow to optimize
            optimization_goals: List of optimization objectives
            performance_issues: Known performance issues
            
        Returns:
            Optimized workflow
        """
        return self.llm_generator.optimize_workflow(
            workflow=workflow,
            performance_issues=performance_issues,
            optimization_goals=optimization_goals
        )
    
    def get_generation_recommendations(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get recommendations for generating a workflow.
        
        Args:
            task_description: Description of the task
            context: Additional context information
            
        Returns:
            Recommendations including suggested mode, parameters, etc.
        """
        context = context or {}
        
        # Analyze task complexity
        from .hybrid_generator import WorkflowComplexityAnalyzer
        complexity_level, complexity_analysis = WorkflowComplexityAnalyzer.analyze_complexity(task_description)
        
        # Find relevant patterns
        relevant_patterns = []
        if self.pattern_store:
            relevant_patterns = self.pattern_store.find_patterns(
                task_type=context.get("task_type"),
                keywords=self.pattern_matcher._extract_keywords(task_description),
                domain=context.get("domain"),
                complexity_level=complexity_level,
                limit=3
            )
        
        # Recommend generation mode
        recommended_mode = self._auto_select_mode(
            task_description, 
            complexity_level,
            context.get("budget_constraint"),
            context.get("latency_requirement", "normal")
        )
        
        # Model recommendations (from Layer 1 integration)
        recommended_model = self.llm_client_factory.select_best_model(
            task_type=context.get("task_type", "workflow_generation"),
            complexity_level=complexity_level,
            budget_constraint=context.get("budget_constraint"),
            latency_requirement=context.get("latency_requirement", "normal")
        )
        
        return {
            "recommended_mode": recommended_mode.value,
            "complexity_analysis": {
                "level": complexity_level,
                "details": complexity_analysis
            },
            "relevant_patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "confidence_score": p.confidence_score,
                    "success_rate": p.success_rate
                } for p in relevant_patterns
            ],
            "recommended_model": recommended_model,
            "estimated_cost": self.llm_client_factory.estimate_cost_for_model(
                recommended_model, task_description, 2000
            ),
            "generation_strategy": self._suggest_generation_strategy(complexity_level, context)
        }
    
    def _auto_select_mode(
        self,
        task_description: str,
        complexity_hint: Optional[str],
        budget_constraint: Optional[float],
        latency_requirement: str
    ) -> GenerationMode:
        """Automatically select the best generation mode"""
        
        # Fast mode for simple tasks or strict latency requirements
        if latency_requirement == "fast" or complexity_hint == "simple":
            return GenerationMode.FAST
        
        # Learning mode if patterns are available and enabled
        if self.enable_learning and self.pattern_store:
            # Check if we have good patterns for this type of task
            keywords = self.pattern_matcher._extract_keywords(task_description)
            patterns = self.pattern_store.find_patterns(keywords=list(keywords), limit=1)
            if patterns and patterns[0].confidence_score > 0.8:
                return GenerationMode.LEARNING
        
        # Creative mode for complex tasks or when explicitly requested
        if complexity_hint == "complex":
            return GenerationMode.CREATIVE
        
        # Default to balanced mode
        return GenerationMode.BALANCED
    
    def _try_pattern_based_generation(
        self,
        task_description: str,
        task_type: Optional[str],
        domain: Optional[str],
        complexity_hint: Optional[str]
    ) -> tuple[Optional[WorkflowGraph], Optional[Any]]:
        """Try to generate workflow using existing patterns"""
        
        # Find relevant patterns
        keywords = self.pattern_matcher._extract_keywords(task_description)
        patterns = self.pattern_store.find_patterns(
            task_type=task_type,
            keywords=list(keywords),
            domain=domain,
            complexity_level=complexity_hint,
            min_confidence=0.7,  # High confidence threshold
            limit=3
        )
        
        if not patterns:
            return None, None
        
        # Use the best pattern to generate a workflow
        best_pattern = patterns[0]
        
        # Convert pattern to workflow (simplified - would need more sophisticated conversion)
        workflow = self._pattern_to_workflow(best_pattern, task_description)
        
        return workflow, best_pattern
    
    def _pattern_to_workflow(self, pattern, task_description: str) -> WorkflowGraph:
        """Convert a pattern to an actual workflow (simplified implementation)"""
        
        # This is a simplified implementation
        # In practice, this would be more sophisticated pattern instantiation
        workflow = WorkflowGraph(
            id=str(uuid.uuid4()),
            name=f"Pattern-based: {pattern.name}",
            description=f"Generated from pattern for: {task_description}",
            metadata={
                "generation_method": "pattern_based",
                "source_pattern": pattern.id,
                "pattern_confidence": pattern.confidence_score
            }
        )
        
        # Use template generator to create a similar structure
        # (This is a fallback - ideally patterns would store more detailed structure)
        template_workflow = self.template_generator.generate_workflow_auto(
            task_description=task_description,
            complexity_hint=pattern.complexity_level
        )
        
        # Copy structure from template workflow
        for node in template_workflow.nodes.values():
            workflow.add_node(node)
        
        for edge in template_workflow.edges:
            workflow.connect(edge.from_node, edge.to_node, edge.data_key, edge.condition)
        
        return workflow
    
    def _generate_with_mode(
        self,
        mode: GenerationMode,
        task_description: str,
        **kwargs
    ) -> WorkflowGraph:
        """Generate workflow using the specified mode"""
        
        if mode == GenerationMode.FAST:
            return self.template_generator.generate_workflow_auto(
                task_description=task_description,
                complexity_hint=kwargs.get("complexity_hint"),
                additional_params=kwargs
            )
        
        elif mode == GenerationMode.CREATIVE:
            return self.llm_generator.generate_workflow(
                task_description=task_description,
                complexity_level=kwargs.get("complexity_hint", "medium"),
                available_tools=kwargs.get("available_tools"),
                model_constraints=kwargs.get("model_constraints"),
                max_iterations=kwargs.get("max_iterations", 3)
            )
        
        elif mode == GenerationMode.LEARNING:
            # Use pattern-based generation with fallback to hybrid
            workflow, _ = self._try_pattern_based_generation(
                task_description,
                kwargs.get("task_type"),
                kwargs.get("domain"),
                kwargs.get("complexity_hint")
            )
            
            if workflow:
                return workflow
            else:
                # Fallback to hybrid
                return self.hybrid_generator.generate_workflow(
                    task_description=task_description,
                    strategy=GenerationStrategy.ADAPTIVE,
                    complexity_hint=kwargs.get("complexity_hint"),
                    **kwargs
                )
        
        else:  # BALANCED or default
            return self.hybrid_generator.generate_workflow(
                task_description=task_description,
                strategy=GenerationStrategy.ADAPTIVE,
                complexity_hint=kwargs.get("complexity_hint"),
                **kwargs
            )
    
    def _learn_from_execution(
        self,
        workflow: WorkflowGraph,
        execution_results: Dict[str, Any],
        **kwargs
    ):
        """Learn patterns from successful workflow execution"""
        
        if not self.pattern_learner:
            return
        
        # Extract learning parameters
        task_description = kwargs.get("task_description", "")
        task_type = kwargs.get("task_type", "general")
        domain = kwargs.get("domain")
        
        # Learn from the execution
        self.pattern_learner.learn_from_workflow(
            workflow=workflow,
            execution_result=execution_results["performance_metrics"],
            task_description=task_description,
            task_type=task_type,
            domain=domain
        )
    
    def _calculate_performance_metrics(self, execution_state) -> Dict[str, Any]:
        """Calculate performance metrics from execution state"""
        
        summary = execution_state.get_execution_summary()
        
        # Calculate basic metrics
        success = summary["status"] == "completed"
        execution_time_ms = summary.get("total_execution_time_ms", 0.0)
        
        # Calculate performance score (simplified)
        performance_score = 10.0 if success else 0.0
        if success and execution_time_ms > 0:
            # Adjust score based on execution time (faster is better)
            if execution_time_ms < 5000:  # Less than 5 seconds
                performance_score = 10.0
            elif execution_time_ms < 30000:  # Less than 30 seconds
                performance_score = 8.0
            elif execution_time_ms < 60000:  # Less than 1 minute
                performance_score = 6.0
            else:
                performance_score = 4.0
        
        return {
            "success": success,
            "performance_score": performance_score,
            "execution_time_ms": execution_time_ms,
            "cost_usd": 0.0,  # Would be calculated from LLM usage
            "error_count": summary.get("failed_nodes", 0),
            "completed_nodes": summary.get("completed_nodes", 0)
        }
    
    def _create_fallback_workflow(self, task_description: str, error_message: str) -> WorkflowGraph:
        """Create a basic fallback workflow when generation fails"""
        
        return self.hybrid_generator._generate_fallback(task_description)
    
    def _update_avg_generation_time(self, generation_time_ms: float):
        """Update average generation time statistics"""
        
        total_generations = self.generation_stats["total_generations"]
        current_avg = self.generation_stats["avg_generation_time_ms"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_generations - 1)) + generation_time_ms) / total_generations
        self.generation_stats["avg_generation_time_ms"] = new_avg
    
    def _suggest_generation_strategy(self, complexity_level: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest generation strategy based on task characteristics"""
        
        strategy = {
            "approach": "hybrid",
            "reasoning": [],
            "recommendations": []
        }
        
        # Analyze context and provide recommendations
        if complexity_level == "simple":
            strategy["approach"] = "template"
            strategy["reasoning"].append("Simple task detected - template approach will be faster and reliable")
            strategy["recommendations"].append("Use pre-defined templates for quick generation")
        
        elif complexity_level == "complex":
            strategy["approach"] = "llm_heavy"
            strategy["reasoning"].append("Complex task detected - LLM generation needed for flexibility")
            strategy["recommendations"].append("Use advanced prompting techniques (CoT, self-refinement)")
        
        else:
            strategy["reasoning"].append("Medium complexity - hybrid approach balances speed and flexibility")
            strategy["recommendations"].append("Start with templates, enhance with LLM if needed")
        
        # Add budget considerations
        if context.get("budget_constraint"):
            strategy["recommendations"].append("Consider cost-efficient models for budget constraints")
        
        # Add latency considerations
        if context.get("latency_requirement") == "fast":
            strategy["recommendations"].append("Prioritize template-based generation for speed")
        
        return strategy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the generator"""
        
        stats = {
            "generation_stats": self.generation_stats.copy(),
            "template_stats": self.template_generator.list_available_templates(),
            "hybrid_stats": self.hybrid_generator.get_generation_stats(),
            "llm_client_stats": self.llm_client_factory.get_client_stats()
        }
        
        # Add pattern statistics if available
        if self.pattern_store:
            stats["pattern_stats"] = self.pattern_store.get_pattern_stats()
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "mode_usage": {mode.value: 0 for mode in GenerationMode},
            "avg_generation_time_ms": 0.0
        }
        
        self.hybrid_generator.reset_stats()
        self.llm_client_factory.reset_all_stats()