"""
LLM-based direct workflow generation implementation
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

from ..workflow.node import WorkflowNode, NodeType
from ..workflow.graph import WorkflowGraph
from .prompts import WorkflowGenerationPrompts
from ..llm_client.client_factory import LLMClientFactory
from ..llm_client.llm_client import LLMRequest


class LLMDirectGenerator:
    """
    Generates workflows by directly prompting LLMs to create workflow specifications.
    
    This approach is more flexible than templates but requires careful prompt
    engineering to ensure consistent, high-quality outputs.
    """
    
    def __init__(self, llm_client_factory: Optional[LLMClientFactory] = None):
        """
        Initialize the LLM Direct Generator.
        
        Args:
            llm_client_factory: Factory for creating LLM clients. If None, creates a new one.
        """
        self.llm_client_factory = llm_client_factory or LLMClientFactory()
        self.prompts = WorkflowGenerationPrompts()
        
        # Default model for workflow generation (can be overridden)
        self.default_model = "gpt-4-turbo"
    
    def generate_workflow(
        self,
        task_description: str,
        complexity_level: str = "medium",
        available_tools: Optional[List[str]] = None,
        model_constraints: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3
    ) -> WorkflowGraph:
        """
        Generate a workflow by prompting an LLM.
        
        Args:
            task_description: Description of the task to create a workflow for
            complexity_level: Expected complexity ("simple", "medium", "complex")
            available_tools: List of available tools
            model_constraints: Constraints on model usage
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Generated WorkflowGraph
        """
        
        # Step 1: Analyze the task
        analysis_prompt = self.prompts.get_workflow_analysis_prompt(
            task_description=task_description,
            complexity_level=complexity_level,
            available_tools=available_tools,
            model_constraints=model_constraints
        )
        
        workflow_analysis = self._call_llm(analysis_prompt)
        
        # Step 2: Generate initial workflow
        generation_prompt = self.prompts.get_workflow_generation_prompt(
            task_description=task_description,
            workflow_analysis=workflow_analysis
        )
        
        workflow_spec = self._call_llm(generation_prompt)
        
        # Step 3: Parse and validate workflow
        workflow = None
        iteration = 0
        
        while iteration < max_iterations:
            try:
                workflow = self._parse_workflow_specification(workflow_spec, task_description)
                
                # Validate the workflow
                is_valid, errors = workflow.validate()
                
                if is_valid:
                    break
                else:
                    # Refine based on validation errors
                    feedback = f"Validation errors: {'; '.join(errors)}"
                    refinement_prompt = self.prompts.get_workflow_refinement_prompt(
                        task_description=task_description,
                        current_workflow=workflow_spec,
                        feedback=feedback
                    )
                    
                    workflow_spec = self._call_llm(refinement_prompt)
                    iteration += 1
                    
            except Exception as e:
                # Handle parsing errors
                feedback = f"Parsing error: {str(e)}"
                refinement_prompt = self.prompts.get_workflow_refinement_prompt(
                    task_description=task_description,
                    current_workflow=workflow_spec,
                    feedback=feedback
                )
                
                workflow_spec = self._call_llm(refinement_prompt)
                iteration += 1
        
        if workflow is None:
            # Fallback: create a simple workflow if LLM generation fails
            workflow = self._create_fallback_workflow(task_description)
        
        return workflow
    
    def optimize_workflow(
        self,
        workflow: WorkflowGraph,
        performance_issues: Optional[List[str]] = None,
        optimization_goals: Optional[List[str]] = None
    ) -> WorkflowGraph:
        """
        Optimize an existing workflow using LLM suggestions.
        
        Args:
            workflow: Existing workflow to optimize
            performance_issues: Known performance issues
            optimization_goals: Optimization objectives
            
        Returns:
            Optimized WorkflowGraph
        """
        # Convert workflow to specification
        current_spec = self._workflow_to_specification(workflow)
        
        # Get optimization prompt
        optimization_prompt = self.prompts.get_workflow_optimization_prompt(
            workflow_specification=json.dumps(current_spec, indent=2),
            performance_issues=performance_issues,
            optimization_goals=optimization_goals
        )
        
        # Get LLM suggestions
        optimization_response = self._call_llm(optimization_prompt)
        
        try:
            # Extract optimized workflow from response
            optimized_spec = self._extract_json_from_response(optimization_response)
            optimized_workflow = self._parse_workflow_specification(
                json.dumps(optimized_spec), 
                workflow.description
            )
            
            return optimized_workflow
            
        except Exception as e:
            # Return original workflow if optimization fails
            print(f"Optimization failed: {e}")
            return workflow
    
    def generate_node(
        self,
        node_purpose: str,
        input_data: str,
        expected_output: str,
        preferred_node_type: str = "LLM_CALL",
        workflow_context: str = ""
    ) -> WorkflowNode:
        """
        Generate a single workflow node using LLM.
        
        Args:
            node_purpose: What this node should accomplish
            input_data: Description of input data
            expected_output: Description of expected output
            preferred_node_type: Preferred type of node
            workflow_context: Context of the overall workflow
            
        Returns:
            Generated WorkflowNode
        """
        node_prompt = self.prompts.get_node_generation_prompt(
            node_purpose=node_purpose,
            input_data=input_data,
            expected_output=expected_output,
            preferred_node_type=preferred_node_type,
            workflow_context=workflow_context
        )
        
        node_response = self._call_llm(node_prompt)
        
        try:
            node_spec = self._extract_json_from_response(node_response)
            return self._create_node_from_spec(node_spec)
        except Exception as e:
            # Fallback: create a basic node
            return self._create_fallback_node(node_purpose, preferred_node_type)
    
    def _call_llm(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        use_cot: bool = False,
        use_self_refine: bool = False,
        use_meta_prompting: bool = False,
        **kwargs
    ) -> str:
        """
        Call the LLM with the given prompt using the integrated LLM client.
        
        This integrates with Layer 1 (Model Selection) and Layer 2 (Prompt Preprocessing).
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model to use (if None, uses default or auto-selects)
            use_cot: Whether to apply Chain-of-Thought prompting
            use_self_refine: Whether to apply self-refinement
            use_meta_prompting: Whether to apply meta-prompting
            **kwargs: Additional parameters for the LLM request
            
        Returns:
            LLM response content
        """
        try:
            # Select model if not specified
            if model is None:
                # Use Layer 1 model selection
                model = self.llm_client_factory.select_best_model(
                    task_type="workflow_generation",
                    complexity_level=kwargs.get("complexity_level", "medium"),
                    budget_constraint=kwargs.get("budget_constraint"),
                    latency_requirement=kwargs.get("latency_requirement", "normal")
                )
            
            # Get appropriate LLM client
            client = self.llm_client_factory.get_client_for_model(model)
            
            # Create LLM request with Layer 2 prompt processing options
            request = LLMRequest(
                prompt=prompt,
                model=model,
                temperature=kwargs.get("temperature", 0.3),  # Lower for more consistent code generation
                max_tokens=kwargs.get("max_tokens", 4000),
                use_cot=use_cot,
                use_self_refine=use_self_refine,
                meta_prompt=use_meta_prompting,
                task_type="workflow_generation",
                complexity_level=kwargs.get("complexity_level", "medium"),
                metadata=kwargs.get("metadata", {})
            )
            
            # Make the LLM call
            response = client.complete_sync(request)
            
            return response.content
            
        except Exception:
            # Fallback to mock response if LLM call fails
            print("LLM call failed, falling back to mock response")
            return self._get_mock_response(prompt)
    
    def _get_mock_response(self, prompt: str) -> str:
        """
        Generate mock responses for testing purposes.
        
        In production, this would be replaced with actual LLM calls.
        """
        if "WORKFLOW ANALYSIS" in prompt:
            return """
WORKFLOW ANALYSIS:
- Task Type: Question & Answer with Research
- Estimated Complexity: medium
- Number of Steps: 4
- Parallel Processing: no

STEP BREAKDOWN:
1. Input Processing: Receive and validate user question - [Node Type: INPUT]
2. Research Planning: Determine search strategy - [Node Type: LLM_CALL]
3. Information Gathering: Search for relevant information - [Node Type: TOOL_USE]
4. Answer Generation: Synthesize final answer - [Node Type: LLM_CALL]
5. Output Delivery: Return formatted response - [Node Type: OUTPUT]

DATA FLOW:
- Input: User question string
- Step 1 → Step 2: Validated question
- Step 2 → Step 3: Search strategy and keywords
- Step 3 → Step 4: Retrieved information
- Step 4 → Step 5: Generated answer
- Output: Formatted response with sources

CONDITIONAL LOGIC:
None required for this workflow

SPECIAL CONSIDERATIONS:
- Handle cases where no information is found
- Ensure answer quality and accuracy
- Include source attribution
"""
        
        elif "Generate a workflow specification" in prompt:
            return """
{
  "workflow_name": "Research-Based Question Answering",
  "workflow_description": "A workflow that researches and answers user questions with source attribution",
  "nodes": [
    {
      "id": "input_question",
      "name": "Question Input",
      "node_type": "INPUT",
      "operation": "receive_user_question",
      "inputs": {},
      "outputs": {"question": "user question string"},
      "config": {}
    },
    {
      "id": "plan_research",
      "name": "Research Planning",
      "node_type": "LLM_CALL",
      "operation": "plan_research_strategy",
      "model_name": "gpt-4",
      "prompt_template": "Create a research strategy for this question: {question}. Provide search keywords and approach.",
      "inputs": {"question": "user question"},
      "outputs": {"search_strategy": "research plan", "keywords": "search terms"},
      "config": {}
    },
    {
      "id": "gather_info",
      "name": "Information Gathering",
      "node_type": "TOOL_USE",
      "operation": "search_information",
      "tool_name": "web_search",
      "tool_params": {"max_results": 10, "search_type": "comprehensive"},
      "inputs": {"keywords": "search terms"},
      "outputs": {"search_results": "found information", "sources": "source URLs"},
      "config": {}
    },
    {
      "id": "generate_answer",
      "name": "Answer Generation",
      "node_type": "LLM_CALL",
      "operation": "synthesize_answer",
      "model_name": "gpt-4",
      "prompt_template": "Based on this information: {search_results}, provide a comprehensive answer to: {question}. Include source references.",
      "inputs": {"question": "original question", "search_results": "research findings"},
      "outputs": {"answer": "generated answer", "confidence": "confidence score"},
      "config": {}
    },
    {
      "id": "format_output",
      "name": "Response Formatting",
      "node_type": "OUTPUT",
      "operation": "format_final_response",
      "inputs": {"answer": "generated answer", "sources": "source references"},
      "outputs": {"formatted_response": "final formatted answer"},
      "config": {}
    }
  ],
  "edges": [
    {"from": "input_question", "to": "plan_research", "data_key": "question"},
    {"from": "plan_research", "to": "gather_info", "data_key": "keywords"},
    {"from": "gather_info", "to": "generate_answer", "data_key": "search_results"},
    {"from": "generate_answer", "to": "format_output", "data_key": "answer"}
  ],
  "start_node": "input_question",
  "end_nodes": ["format_output"],
  "metadata": {
    "complexity": "medium",
    "estimated_time": "30-60 seconds",
    "resource_requirements": ["web_search", "gpt-4"]
  }
}
"""
        
        else:
            return "Mock LLM response for prompt: " + prompt[:100] + "..."
    
    def _parse_workflow_specification(self, workflow_spec: str, task_description: str) -> WorkflowGraph:
        """Parse a JSON workflow specification into a WorkflowGraph"""
        
        # Extract JSON from the response
        spec_dict = self._extract_json_from_response(workflow_spec)
        
        # Create workflow
        workflow = WorkflowGraph(
            id=str(uuid.uuid4()),
            name=spec_dict.get("workflow_name", "LLM Generated Workflow"),
            description=spec_dict.get("workflow_description", task_description),
            metadata={
                "generation_method": "llm_direct",
                "generated_at": datetime.now().isoformat(),
                "original_task": task_description,
                **spec_dict.get("metadata", {})
            }
        )
        
        # Create nodes
        for node_spec in spec_dict.get("nodes", []):
            node = self._create_node_from_spec(node_spec)
            workflow.add_node(node)
        
        # Create edges
        for edge_spec in spec_dict.get("edges", []):
            workflow.connect(
                from_node_id=edge_spec["from"],
                to_node_id=edge_spec["to"],
                data_key=edge_spec["data_key"],
                condition=edge_spec.get("condition")
            )
        
        # Set start and end nodes
        if "start_node" in spec_dict:
            workflow.start_node = spec_dict["start_node"]
        
        if "end_nodes" in spec_dict:
            workflow.end_nodes = spec_dict["end_nodes"]
        
        return workflow
    
    def _create_node_from_spec(self, node_spec: Dict[str, Any]) -> WorkflowNode:
        """Create a WorkflowNode from a specification dictionary"""
        
        node = WorkflowNode(
            id=node_spec["id"],
            name=node_spec["name"],
            node_type=NodeType(node_spec["node_type"]),
            operation=node_spec["operation"],
            inputs=node_spec.get("inputs", {}),
            outputs=node_spec.get("outputs", {}),
            config=node_spec.get("config", {})
        )
        
        # Set type-specific fields
        if "model_name" in node_spec:
            node.model_name = node_spec["model_name"]
        
        if "prompt_template" in node_spec:
            node.prompt_template = node_spec["prompt_template"]
        
        if "tool_name" in node_spec:
            node.tool_name = node_spec["tool_name"]
        
        if "tool_params" in node_spec:
            node.tool_params = node_spec["tool_params"]
        
        if "condition" in node_spec:
            node.condition = node_spec["condition"]
        
        if "branches" in node_spec:
            node.branches = node_spec["branches"]
        
        return node
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response that might contain additional text"""
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If direct JSON parsing fails, try to clean the response
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from LLM response: {e}")
    
    def _workflow_to_specification(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """Convert a WorkflowGraph back to specification format"""
        
        nodes = []
        for node_id, node in workflow.nodes.items():
            node_spec = {
                "id": node.id,
                "name": node.name,
                "node_type": node.node_type.value,
                "operation": node.operation,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "config": node.config
            }
            
            if node.model_name:
                node_spec["model_name"] = node.model_name
            if node.prompt_template:
                node_spec["prompt_template"] = node.prompt_template
            if node.tool_name:
                node_spec["tool_name"] = node.tool_name
            if node.tool_params:
                node_spec["tool_params"] = node.tool_params
            if node.condition:
                node_spec["condition"] = node.condition
            if node.branches:
                node_spec["branches"] = node.branches
            
            nodes.append(node_spec)
        
        edges = []
        for edge in workflow.edges:
            edge_spec = {
                "from": edge.from_node,
                "to": edge.to_node,
                "data_key": edge.data_key
            }
            if edge.condition:
                edge_spec["condition"] = edge.condition
            edges.append(edge_spec)
        
        return {
            "workflow_name": workflow.name,
            "workflow_description": workflow.description,
            "nodes": nodes,
            "edges": edges,
            "start_node": workflow.start_node,
            "end_nodes": workflow.end_nodes,
            "metadata": workflow.metadata
        }
    
    def _create_fallback_workflow(self, task_description: str) -> WorkflowGraph:
        """Create a simple fallback workflow when LLM generation fails"""
        
        workflow = WorkflowGraph(
            id=str(uuid.uuid4()),
            name="Fallback Simple Workflow",
            description=f"Fallback workflow for: {task_description}",
            metadata={"generation_method": "fallback", "original_task": task_description}
        )
        
        # Create simple input -> process -> output workflow
        input_node = WorkflowNode(
            id="input",
            name="Input",
            node_type=NodeType.INPUT,
            operation="receive_input"
        )
        
        process_node = WorkflowNode(
            id="process",
            name="Process",
            node_type=NodeType.LLM_CALL,
            operation="process_task",
            model_name="gpt-4",
            prompt_template=f"Complete this task: {task_description}. Input: {{input_data}}"
        )
        
        output_node = WorkflowNode(
            id="output",
            name="Output",
            node_type=NodeType.OUTPUT,
            operation="return_result"
        )
        
        workflow.add_node(input_node)
        workflow.add_node(process_node)
        workflow.add_node(output_node)
        
        workflow.connect("input", "process", "input_data")
        workflow.connect("process", "output", "result")
        
        return workflow
    
    def _create_fallback_node(self, purpose: str, node_type: str) -> WorkflowNode:
        """Create a fallback node when LLM generation fails"""
        
        return WorkflowNode(
            id=f"fallback_{uuid.uuid4().hex[:8]}",
            name=f"Fallback {purpose}",
            node_type=NodeType(node_type) if node_type in [nt.value for nt in NodeType] else NodeType.LLM_CALL,
            operation=purpose,
            model_name="gpt-4" if node_type == "LLM_CALL" else None,
            prompt_template=f"Handle: {purpose}. Input: {{input}}" if node_type == "LLM_CALL" else None
        )