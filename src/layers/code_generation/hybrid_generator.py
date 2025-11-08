"""
Hybrid workflow generator that combines template-based and LLM-direct approaches
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import uuid
from datetime import datetime

from .workflow.graph import WorkflowGraph
from .templates.template_generator import TemplateBasedGenerator
from .llm_generators.llm_generator import LLMDirectGenerator
from .llm_client.client_factory import LLMClientFactory


class GenerationStrategy(str, Enum):
    """Available generation strategies"""
    TEMPLATE_ONLY = "template_only"
    LLM_ONLY = "llm_only"
    HYBRID_TEMPLATE_FIRST = "hybrid_template_first"
    HYBRID_LLM_FIRST = "hybrid_llm_first"
    ADAPTIVE = "adaptive"


class WorkflowComplexityAnalyzer:
    """Analyzes task complexity to choose appropriate generation strategy"""
    
    @staticmethod
    def analyze_complexity(task_description: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze task complexity and characteristics.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple of (complexity_level, analysis_details)
        """
        task_lower = task_description.lower()
        
        # Initialize complexity indicators
        complexity_indicators = {
            "simple_keywords": 0,
            "complex_keywords": 0,
            "workflow_indicators": 0,
            "tool_requirements": 0,
            "reasoning_requirements": 0,
            "length": len(task_description)
        }
        
        # Simple task indicators
        simple_keywords = [
            "answer", "question", "simple", "basic", "single", "direct",
            "quick", "straightforward", "one", "just"
        ]
        for keyword in simple_keywords:
            if keyword in task_lower:
                complexity_indicators["simple_keywords"] += 1
        
        # Complex task indicators
        complex_keywords = [
            "analyze", "research", "multiple", "steps", "complex", "detailed",
            "comprehensive", "integrate", "optimize", "sophisticated", "advanced",
            "multi-step", "elaborate", "thorough"
        ]
        for keyword in complex_keywords:
            if keyword in task_lower:
                complexity_indicators["complex_keywords"] += 1
        
        # Workflow indicators
        workflow_keywords = [
            "workflow", "process", "pipeline", "sequence", "flow", "chain",
            "steps", "stages", "phases", "automation"
        ]
        for keyword in workflow_keywords:
            if keyword in task_lower:
                complexity_indicators["workflow_indicators"] += 1
        
        # Tool requirement indicators
        tool_keywords = [
            "search", "web", "api", "database", "file", "code", "execute",
            "calculate", "generate", "create", "build", "test"
        ]
        for keyword in tool_keywords:
            if keyword in task_lower:
                complexity_indicators["tool_requirements"] += 1
        
        # Reasoning requirement indicators
        reasoning_keywords = [
            "reasoning", "logic", "decide", "choose", "compare", "evaluate",
            "assess", "determine", "judge", "conclude", "infer"
        ]
        for keyword in reasoning_keywords:
            if keyword in task_lower:
                complexity_indicators["reasoning_requirements"] += 1
        
        # Determine complexity level
        simple_score = complexity_indicators["simple_keywords"] * 2
        complex_score = (
            complexity_indicators["complex_keywords"] * 3 +
            complexity_indicators["workflow_indicators"] * 2 +
            complexity_indicators["tool_requirements"] * 1.5 +
            complexity_indicators["reasoning_requirements"] * 2
        )
        
        # Length factor
        if complexity_indicators["length"] > 200:
            complex_score += 2
        elif complexity_indicators["length"] < 50:
            simple_score += 1
        
        # Determine final complexity
        if simple_score > complex_score and simple_score > 3:
            complexity_level = "simple"
        elif complex_score > simple_score * 1.5:
            complexity_level = "complex"
        else:
            complexity_level = "medium"
        
        return complexity_level, complexity_indicators


class HybridWorkflowGenerator:
    """
    Hybrid workflow generator that intelligently combines template-based
    and LLM-direct generation strategies.
    
    This provides the benefits of both approaches:
    - Template reliability and speed for common patterns
    - LLM flexibility for complex or novel requirements
    """
    
    def __init__(self, llm_client_factory: Optional[LLMClientFactory] = None):
        """
        Initialize the hybrid generator.
        
        Args:
            llm_client_factory: Factory for LLM clients
        """
        self.llm_client_factory = llm_client_factory or LLMClientFactory()
        
        # Initialize sub-generators
        self.template_generator = TemplateBasedGenerator()
        self.llm_generator = LLMDirectGenerator(self.llm_client_factory)
        
        # Strategy configuration
        self.default_strategy = GenerationStrategy.ADAPTIVE
        
        # Performance tracking
        self.generation_stats = {
            "template_successes": 0,
            "llm_successes": 0,
            "hybrid_successes": 0,
            "total_generations": 0
        }
    
    def generate_workflow(
        self,
        task_description: str,
        strategy: Optional[GenerationStrategy] = None,
        complexity_hint: Optional[str] = None,
        **kwargs
    ) -> WorkflowGraph:
        """
        Generate a workflow using the hybrid approach.
        
        Args:
            task_description: Description of the task
            strategy: Generation strategy to use (if None, uses adaptive)
            complexity_hint: Manual complexity hint
            **kwargs: Additional parameters
            
        Returns:
            Generated WorkflowGraph
        """
        self.generation_stats["total_generations"] += 1
        
        # Analyze task complexity if not provided
        if complexity_hint is None:
            complexity_level, complexity_analysis = WorkflowComplexityAnalyzer.analyze_complexity(task_description)
        else:
            complexity_level = complexity_hint
            complexity_analysis = {}
        
        # Determine strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(task_description, complexity_level, complexity_analysis)
        
        # Generate workflow based on strategy
        try:
            if strategy == GenerationStrategy.TEMPLATE_ONLY:
                workflow = self._generate_template_only(task_description, complexity_level, **kwargs)
                self.generation_stats["template_successes"] += 1
                
            elif strategy == GenerationStrategy.LLM_ONLY:
                workflow = self._generate_llm_only(task_description, complexity_level, **kwargs)
                self.generation_stats["llm_successes"] += 1
                
            elif strategy == GenerationStrategy.HYBRID_TEMPLATE_FIRST:
                workflow = self._generate_hybrid_template_first(task_description, complexity_level, **kwargs)
                self.generation_stats["hybrid_successes"] += 1
                
            elif strategy == GenerationStrategy.HYBRID_LLM_FIRST:
                workflow = self._generate_hybrid_llm_first(task_description, complexity_level, **kwargs)
                self.generation_stats["hybrid_successes"] += 1
                
            else:  # ADAPTIVE
                workflow = self._generate_adaptive(task_description, complexity_level, complexity_analysis, **kwargs)
                self.generation_stats["hybrid_successes"] += 1
            
            # Add generation metadata
            workflow.metadata.update({
                "generation_strategy": strategy.value,
                "complexity_level": complexity_level,
                "complexity_analysis": complexity_analysis,
                "generated_at": datetime.now().isoformat(),
                "generator": "hybrid"
            })
            
            return workflow
            
        except Exception as e:
            # Fallback to simple template generation
            print(f"Hybrid generation failed: {e}, falling back to template")
            return self._generate_fallback(task_description)
    
    def _select_strategy(
        self, 
        task_description: str, 
        complexity_level: str, 
        complexity_analysis: Dict[str, Any]
    ) -> GenerationStrategy:
        """Select the best generation strategy based on task analysis"""
        
        # Simple tasks - prefer templates
        if complexity_level == "simple":
            # Check if we have a good template match
            recommended_template = self.template_generator.template_library.recommend_template(task_description)
            if recommended_template:
                return GenerationStrategy.TEMPLATE_ONLY
            else:
                return GenerationStrategy.HYBRID_LLM_FIRST
        
        # Complex tasks - prefer LLM with template fallback
        elif complexity_level == "complex":
            return GenerationStrategy.HYBRID_LLM_FIRST
        
        # Medium complexity - use adaptive approach
        else:
            # Check for strong template indicators
            if complexity_analysis.get("workflow_indicators", 0) > 0:
                return GenerationStrategy.HYBRID_TEMPLATE_FIRST
            else:
                return GenerationStrategy.HYBRID_LLM_FIRST
    
    def _generate_template_only(
        self, 
        task_description: str, 
        complexity_level: str, 
        **kwargs
    ) -> WorkflowGraph:
        """Generate workflow using only templates"""
        return self.template_generator.generate_workflow_auto(
            task_description=task_description,
            complexity_hint=complexity_level,
            additional_params=kwargs.get("template_params", {})
        )
    
    def _generate_llm_only(
        self, 
        task_description: str, 
        complexity_level: str, 
        **kwargs
    ) -> WorkflowGraph:
        """Generate workflow using only LLM"""
        return self.llm_generator.generate_workflow(
            task_description=task_description,
            complexity_level=complexity_level,
            available_tools=kwargs.get("available_tools"),
            model_constraints=kwargs.get("model_constraints"),
            max_iterations=kwargs.get("max_iterations", 3)
        )
    
    def _generate_hybrid_template_first(
        self, 
        task_description: str, 
        complexity_level: str, 
        **kwargs
    ) -> WorkflowGraph:
        """Generate using templates first, then LLM enhancement"""
        
        # Step 1: Generate base workflow with template
        try:
            base_workflow = self.template_generator.generate_workflow_auto(
                task_description=task_description,
                complexity_hint=complexity_level,
                additional_params=kwargs.get("template_params", {})
            )
            
            # Step 2: Enhance with LLM if needed
            if self._needs_enhancement(base_workflow, task_description):
                enhanced_workflow = self._enhance_workflow_with_llm(
                    base_workflow, task_description, **kwargs
                )
                return enhanced_workflow
            else:
                return base_workflow
                
        except Exception as e:
            # Fallback to LLM-only
            print(f"Template generation failed: {e}, falling back to LLM")
            return self._generate_llm_only(task_description, complexity_level, **kwargs)
    
    def _generate_hybrid_llm_first(
        self, 
        task_description: str, 
        complexity_level: str, 
        **kwargs
    ) -> WorkflowGraph:
        """Generate using LLM first, with template fallback"""
        
        # Step 1: Try LLM generation
        try:
            llm_workflow = self.llm_generator.generate_workflow(
                task_description=task_description,
                complexity_level=complexity_level,
                available_tools=kwargs.get("available_tools"),
                model_constraints=kwargs.get("model_constraints"),
                max_iterations=kwargs.get("max_iterations", 2)  # Fewer iterations for hybrid
            )
            
            return llm_workflow
            
        except Exception as e:
            # Step 2: Fallback to template
            print(f"LLM generation failed: {e}, falling back to template")
            return self._generate_template_only(task_description, complexity_level, **kwargs)
    
    def _generate_adaptive(
        self, 
        task_description: str, 
        complexity_level: str, 
        complexity_analysis: Dict[str, Any], 
        **kwargs
    ) -> WorkflowGraph:
        """Adaptive generation that dynamically chooses the best approach"""
        
        # Check if we have a good template match
        recommended_template = self.template_generator.template_library.recommend_template(task_description)
        
        if recommended_template and complexity_level in ["simple", "medium"]:
            # Use template-first approach
            return self._generate_hybrid_template_first(task_description, complexity_level, **kwargs)
        else:
            # Use LLM-first approach
            return self._generate_hybrid_llm_first(task_description, complexity_level, **kwargs)
    
    def _needs_enhancement(self, workflow: WorkflowGraph, task_description: str) -> bool:
        """Determine if a template-generated workflow needs LLM enhancement"""
        
        # Check workflow complexity vs task requirements
        workflow_stats = workflow.get_statistics()
        
        # Simple heuristics for enhancement decision
        if workflow_stats["total_nodes"] < 3:
            # Very simple workflow might need enhancement
            return True
        
        if "custom" in task_description.lower() or "specific" in task_description.lower():
            # Custom requirements likely need enhancement
            return True
        
        return False
    
    def _enhance_workflow_with_llm(
        self, 
        base_workflow: WorkflowGraph, 
        task_description: str, 
        **kwargs
    ) -> WorkflowGraph:
        """Enhance a template-generated workflow using LLM"""
        
        # Convert workflow to specification for LLM analysis
        from .llm_generators.llm_generator import LLMDirectGenerator
        
        current_spec = self.llm_generator._workflow_to_specification(base_workflow)
        
        # Use LLM to optimize/enhance the workflow
        enhanced_workflow = self.llm_generator.optimize_workflow(
            base_workflow,
            performance_issues=["limited_functionality"],
            optimization_goals=["better_task_match", "enhanced_capability"]
        )
        
        return enhanced_workflow
    
    def _generate_fallback(self, task_description: str) -> WorkflowGraph:
        """Generate a basic fallback workflow when all else fails"""
        
        workflow = WorkflowGraph(
            id=str(uuid.uuid4()),
            name="Fallback Workflow",
            description=f"Fallback workflow for: {task_description}",
            metadata={"generation_method": "fallback", "original_task": task_description}
        )
        
        # Create minimal workflow structure
        from .workflow.node import WorkflowNode, NodeType
        
        input_node = WorkflowNode(
            id="fallback_input",
            name="Input",
            node_type=NodeType.INPUT,
            operation="receive_input"
        )
        
        process_node = WorkflowNode(
            id="fallback_process",
            name="Process Task",
            node_type=NodeType.LLM_CALL,
            operation="process_user_request",
            model_name="gpt-3.5-turbo",
            prompt_template=f"Handle this request: {task_description}. Input: {{input_data}}"
        )
        
        output_node = WorkflowNode(
            id="fallback_output",
            name="Output",
            node_type=NodeType.OUTPUT,
            operation="return_result"
        )
        
        workflow.add_node(input_node)
        workflow.add_node(process_node)
        workflow.add_node(output_node)
        
        workflow.connect("fallback_input", "fallback_process", "input_data")
        workflow.connect("fallback_process", "fallback_output", "result")
        
        return workflow
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation performance statistics"""
        total = max(self.generation_stats["total_generations"], 1)
        
        return {
            **self.generation_stats,
            "template_success_rate": self.generation_stats["template_successes"] / total,
            "llm_success_rate": self.generation_stats["llm_successes"] / total,
            "hybrid_success_rate": self.generation_stats["hybrid_successes"] / total
        }
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.generation_stats = {
            "template_successes": 0,
            "llm_successes": 0,
            "hybrid_successes": 0,
            "total_generations": 0
        }