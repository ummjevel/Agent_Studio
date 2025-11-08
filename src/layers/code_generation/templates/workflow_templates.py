"""
Pre-defined workflow templates for common use cases
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

from ..workflow.node import WorkflowNode, NodeType
from ..workflow.graph import WorkflowGraph


class TaskType(str, Enum):
    """Common task types for template selection"""
    SIMPLE_QA = "simple_qa"
    RESEARCH_ANALYSIS = "research_analysis"
    CODE_GENERATION = "code_generation"
    DATA_PROCESSING = "data_processing"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    TOOL_CHAIN = "tool_chain"
    DECISION_TREE = "decision_tree"
    PARALLEL_PROCESSING = "parallel_processing"


class WorkflowTemplate(BaseModel):
    """Represents a workflow template with parameterizable components"""
    
    id: str
    name: str
    description: str
    task_type: TaskType
    complexity_level: str  # "simple", "medium", "complex"
    
    # Template structure
    node_templates: List[Dict[str, Any]]
    edge_templates: List[Dict[str, Any]]
    
    # Parameterization
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_params: List[str] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)


class WorkflowTemplateLibrary:
    """Library of pre-defined workflow templates"""
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize the library with common workflow templates"""
        
        # Simple Q&A Template
        self.templates["simple_qa"] = WorkflowTemplate(
            id="simple_qa",
            name="Simple Question & Answer",
            description="Basic single-step Q&A workflow",
            task_type=TaskType.SIMPLE_QA,
            complexity_level="simple",
            node_templates=[
                {
                    "id": "input",
                    "name": "User Input",
                    "node_type": NodeType.INPUT.value,
                    "operation": "receive_question"
                },
                {
                    "id": "llm_qa",
                    "name": "LLM Q&A",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "answer_question",
                    "model_name": "{model_name}",
                    "prompt_template": "Answer the following question: {question}"
                },
                {
                    "id": "output",
                    "name": "Response Output",
                    "node_type": NodeType.OUTPUT.value,
                    "operation": "return_answer"
                }
            ],
            edge_templates=[
                {"from": "input", "to": "llm_qa", "data_key": "question"},
                {"from": "llm_qa", "to": "output", "data_key": "answer"}
            ],
            required_params=["model_name"],
            tags=["qa", "simple", "single-step"],
            use_cases=["Basic Q&A", "Information retrieval", "Simple queries"]
        )
        
        # Research & Analysis Template
        self.templates["research_analysis"] = WorkflowTemplate(
            id="research_analysis",
            name="Research & Analysis",
            description="Multi-step research and analysis workflow",
            task_type=TaskType.RESEARCH_ANALYSIS,
            complexity_level="medium",
            node_templates=[
                {
                    "id": "input",
                    "name": "Research Topic Input",
                    "node_type": NodeType.INPUT.value,
                    "operation": "receive_topic"
                },
                {
                    "id": "search_planning",
                    "name": "Search Strategy Planning",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "plan_search_strategy",
                    "model_name": "{model_name}",
                    "prompt_template": "Create a search strategy for researching: {topic}"
                },
                {
                    "id": "information_gathering",
                    "name": "Information Gathering",
                    "node_type": NodeType.TOOL_USE.value,
                    "operation": "gather_information",
                    "tool_name": "web_search"
                },
                {
                    "id": "analysis",
                    "name": "Information Analysis",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "analyze_information",
                    "model_name": "{model_name}",
                    "prompt_template": "Analyze the following information about {topic}: {gathered_info}"
                },
                {
                    "id": "synthesis",
                    "name": "Synthesis & Summary",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "synthesize_findings",
                    "model_name": "{model_name}",
                    "prompt_template": "Synthesize findings into a comprehensive summary: {analysis}"
                },
                {
                    "id": "output",
                    "name": "Research Report",
                    "node_type": NodeType.OUTPUT.value,
                    "operation": "return_report"
                }
            ],
            edge_templates=[
                {"from": "input", "to": "search_planning", "data_key": "topic"},
                {"from": "search_planning", "to": "information_gathering", "data_key": "search_strategy"},
                {"from": "information_gathering", "to": "analysis", "data_key": "gathered_info"},
                {"from": "analysis", "to": "synthesis", "data_key": "analysis"},
                {"from": "synthesis", "to": "output", "data_key": "report"}
            ],
            required_params=["model_name"],
            tags=["research", "analysis", "multi-step"],
            use_cases=["Market research", "Academic research", "Competitive analysis"]
        )
        
        # Code Generation Template
        self.templates["code_generation"] = WorkflowTemplate(
            id="code_generation",
            name="Code Generation",
            description="Code generation with testing and refinement",
            task_type=TaskType.CODE_GENERATION,
            complexity_level="medium",
            node_templates=[
                {
                    "id": "input",
                    "name": "Requirements Input",
                    "node_type": NodeType.INPUT.value,
                    "operation": "receive_requirements"
                },
                {
                    "id": "code_planning",
                    "name": "Code Planning",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "plan_code_structure",
                    "model_name": "{model_name}",
                    "prompt_template": "Plan the code structure for: {requirements}"
                },
                {
                    "id": "code_generation",
                    "name": "Code Generation",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "generate_code",
                    "model_name": "{model_name}",
                    "prompt_template": "Generate code based on plan: {plan}"
                },
                {
                    "id": "code_testing",
                    "name": "Code Testing",
                    "node_type": NodeType.TOOL_USE.value,
                    "operation": "test_code",
                    "tool_name": "code_executor"
                },
                {
                    "id": "refinement_check",
                    "name": "Refinement Check",
                    "node_type": NodeType.DECISION.value,
                    "operation": "check_if_refinement_needed",
                    "condition": "test_results.success == False"
                },
                {
                    "id": "code_refinement",
                    "name": "Code Refinement",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "refine_code",
                    "model_name": "{model_name}",
                    "prompt_template": "Refine code based on test results: {test_results}"
                },
                {
                    "id": "output",
                    "name": "Final Code",
                    "node_type": NodeType.OUTPUT.value,
                    "operation": "return_code"
                }
            ],
            edge_templates=[
                {"from": "input", "to": "code_planning", "data_key": "requirements"},
                {"from": "code_planning", "to": "code_generation", "data_key": "plan"},
                {"from": "code_generation", "to": "code_testing", "data_key": "code"},
                {"from": "code_testing", "to": "refinement_check", "data_key": "test_results"},
                {"from": "refinement_check", "to": "code_refinement", "data_key": "test_results", 
                 "condition": "test_results.success == False"},
                {"from": "refinement_check", "to": "output", "data_key": "code", 
                 "condition": "test_results.success == True"},
                {"from": "code_refinement", "to": "code_testing", "data_key": "refined_code"}
            ],
            required_params=["model_name"],
            tags=["code", "generation", "testing", "refinement"],
            use_cases=["Code generation", "Programming assistance", "Automated development"]
        )
        
        # Multi-step Reasoning Template
        self.templates["multi_step_reasoning"] = WorkflowTemplate(
            id="multi_step_reasoning",
            name="Multi-Step Reasoning",
            description="Chain-of-thought reasoning workflow",
            task_type=TaskType.MULTI_STEP_REASONING,
            complexity_level="medium",
            node_templates=[
                {
                    "id": "input",
                    "name": "Problem Input",
                    "node_type": NodeType.INPUT.value,
                    "operation": "receive_problem"
                },
                {
                    "id": "problem_decomposition",
                    "name": "Problem Decomposition",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "decompose_problem",
                    "model_name": "{model_name}",
                    "prompt_template": "Break down this problem into steps: {problem}"
                },
                {
                    "id": "step_1",
                    "name": "Reasoning Step 1",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "reasoning_step_1",
                    "model_name": "{model_name}",
                    "prompt_template": "Solve step 1: {step_1_description}"
                },
                {
                    "id": "step_2",
                    "name": "Reasoning Step 2",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "reasoning_step_2",
                    "model_name": "{model_name}",
                    "prompt_template": "Solve step 2 using results from step 1: {step_1_result}, {step_2_description}"
                },
                {
                    "id": "step_3",
                    "name": "Reasoning Step 3",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "reasoning_step_3",
                    "model_name": "{model_name}",
                    "prompt_template": "Solve step 3 using previous results: {step_2_result}, {step_3_description}"
                },
                {
                    "id": "synthesis",
                    "name": "Solution Synthesis",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "synthesize_solution",
                    "model_name": "{model_name}",
                    "prompt_template": "Combine all steps into final solution: {all_steps}"
                },
                {
                    "id": "output",
                    "name": "Final Solution",
                    "node_type": NodeType.OUTPUT.value,
                    "operation": "return_solution"
                }
            ],
            edge_templates=[
                {"from": "input", "to": "problem_decomposition", "data_key": "problem"},
                {"from": "problem_decomposition", "to": "step_1", "data_key": "step_1_description"},
                {"from": "step_1", "to": "step_2", "data_key": "step_1_result"},
                {"from": "step_2", "to": "step_3", "data_key": "step_2_result"},
                {"from": "step_3", "to": "synthesis", "data_key": "step_3_result"},
                {"from": "synthesis", "to": "output", "data_key": "solution"}
            ],
            required_params=["model_name"],
            tags=["reasoning", "chain-of-thought", "problem-solving"],
            use_cases=["Complex problem solving", "Mathematical reasoning", "Logical analysis"]
        )
        
        # Parallel Processing Template
        self.templates["parallel_processing"] = WorkflowTemplate(
            id="parallel_processing",
            name="Parallel Processing",
            description="Process multiple items in parallel",
            task_type=TaskType.PARALLEL_PROCESSING,
            complexity_level="complex",
            node_templates=[
                {
                    "id": "input",
                    "name": "Data Input",
                    "node_type": NodeType.INPUT.value,
                    "operation": "receive_data_list"
                },
                {
                    "id": "splitter",
                    "name": "Data Splitter",
                    "node_type": NodeType.TRANSFORM.value,
                    "operation": "split_data_for_parallel_processing"
                },
                {
                    "id": "parallel_processor_1",
                    "name": "Parallel Processor 1",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "process_chunk_1",
                    "model_name": "{model_name}",
                    "prompt_template": "Process this data chunk: {chunk_1}"
                },
                {
                    "id": "parallel_processor_2",
                    "name": "Parallel Processor 2",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "process_chunk_2",
                    "model_name": "{model_name}",
                    "prompt_template": "Process this data chunk: {chunk_2}"
                },
                {
                    "id": "parallel_processor_3",
                    "name": "Parallel Processor 3",
                    "node_type": NodeType.LLM_CALL.value,
                    "operation": "process_chunk_3",
                    "model_name": "{model_name}",
                    "prompt_template": "Process this data chunk: {chunk_3}"
                },
                {
                    "id": "aggregator",
                    "name": "Result Aggregator",
                    "node_type": NodeType.TRANSFORM.value,
                    "operation": "aggregate_parallel_results"
                },
                {
                    "id": "output",
                    "name": "Aggregated Results",
                    "node_type": NodeType.OUTPUT.value,
                    "operation": "return_aggregated_results"
                }
            ],
            edge_templates=[
                {"from": "input", "to": "splitter", "data_key": "data_list"},
                {"from": "splitter", "to": "parallel_processor_1", "data_key": "chunk_1"},
                {"from": "splitter", "to": "parallel_processor_2", "data_key": "chunk_2"},
                {"from": "splitter", "to": "parallel_processor_3", "data_key": "chunk_3"},
                {"from": "parallel_processor_1", "to": "aggregator", "data_key": "result_1"},
                {"from": "parallel_processor_2", "to": "aggregator", "data_key": "result_2"},
                {"from": "parallel_processor_3", "to": "aggregator", "data_key": "result_3"},
                {"from": "aggregator", "to": "output", "data_key": "aggregated_results"}
            ],
            required_params=["model_name"],
            tags=["parallel", "processing", "scalable"],
            use_cases=["Batch processing", "Parallel analysis", "Scalable workflows"]
        )
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_task_type(self, task_type: TaskType) -> List[WorkflowTemplate]:
        """Get all templates for a specific task type"""
        return [template for template in self.templates.values() 
                if template.task_type == task_type]
    
    def get_templates_by_complexity(self, complexity: str) -> List[WorkflowTemplate]:
        """Get all templates for a specific complexity level"""
        return [template for template in self.templates.values() 
                if template.complexity_level == complexity]
    
    def list_all_templates(self) -> List[WorkflowTemplate]:
        """Get all available templates"""
        return list(self.templates.values())
    
    def add_template(self, template: WorkflowTemplate) -> None:
        """Add a new template to the library"""
        self.templates[template.id] = template
    
    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the library"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def find_templates_by_tags(self, tags: List[str]) -> List[WorkflowTemplate]:
        """Find templates that contain any of the specified tags"""
        matching_templates = []
        for template in self.templates.values():
            if any(tag in template.tags for tag in tags):
                matching_templates.append(template)
        return matching_templates
    
    def recommend_template(self, task_description: str, complexity_hint: Optional[str] = None) -> Optional[WorkflowTemplate]:
        """
        Recommend a template based on task description.
        This is a simple implementation - could be enhanced with ML in the future.
        """
        task_description_lower = task_description.lower()
        
        # Simple keyword matching
        if any(keyword in task_description_lower for keyword in ["question", "answer", "qa", "ask"]):
            return self.get_template("simple_qa")
        elif any(keyword in task_description_lower for keyword in ["research", "analyze", "study", "investigate"]):
            return self.get_template("research_analysis")
        elif any(keyword in task_description_lower for keyword in ["code", "program", "develop", "implement"]):
            return self.get_template("code_generation")
        elif any(keyword in task_description_lower for keyword in ["reasoning", "solve", "problem", "steps"]):
            return self.get_template("multi_step_reasoning")
        elif any(keyword in task_description_lower for keyword in ["parallel", "batch", "multiple", "concurrent"]):
            return self.get_template("parallel_processing")
        
        # Default to simple Q&A if no match
        return self.get_template("simple_qa")