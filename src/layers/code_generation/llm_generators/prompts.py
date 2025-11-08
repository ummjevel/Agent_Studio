"""
Prompts for LLM-based workflow generation
"""

from typing import Dict, Any, List, Optional


class WorkflowGenerationPrompts:
    """
    Collection of prompts for generating workflows using LLMs.
    
    These prompts are designed to elicit structured workflow descriptions
    that can be parsed into WorkflowGraph objects.
    """
    
    WORKFLOW_ANALYSIS_PROMPT = """
You are an expert workflow designer. Analyze the given task and break it down into a structured workflow.

Task: {task_description}

Consider the following:
1. What are the main steps needed to complete this task?
2. What type of processing is needed at each step? (LLM reasoning, tool usage, data transformation, decision making)
3. What data flows between steps?
4. Are there any conditional branches or parallel processing opportunities?
5. What are the inputs and outputs of the overall workflow?

Complexity Level: {complexity_level}
Available Tools: {available_tools}
Model Constraints: {model_constraints}

Please provide your analysis in the following structured format:

WORKFLOW ANALYSIS:
- Task Type: [type of task]
- Estimated Complexity: [simple/medium/complex]
- Number of Steps: [number]
- Parallel Processing: [yes/no]

STEP BREAKDOWN:
1. [Step Name]: [Description] - [Node Type: LLM_CALL/TOOL_USE/DECISION/TRANSFORM/etc.]
2. [Step Name]: [Description] - [Node Type]
...

DATA FLOW:
- Input: [what data comes into the workflow]
- Step 1 → Step 2: [what data passes between steps]
- Step 2 → Step 3: [what data passes between steps]
...
- Output: [what data comes out of the workflow]

CONDITIONAL LOGIC:
[If any decision points or branches exist, describe them]

SPECIAL CONSIDERATIONS:
[Any specific requirements, error handling, or optimization opportunities]
"""

    WORKFLOW_GENERATION_PROMPT = """
Based on your analysis, generate a detailed workflow specification.

Task: {task_description}
Analysis: {workflow_analysis}

Generate a workflow specification in the following JSON format:

{{
  "workflow_name": "descriptive name for the workflow",
  "workflow_description": "detailed description of what this workflow does",
  "nodes": [
    {{
      "id": "unique_node_id",
      "name": "Human readable name",
      "node_type": "LLM_CALL|TOOL_USE|DECISION|TRANSFORM|INPUT|OUTPUT|CONDITION|PARALLEL|SEQUENTIAL",
      "operation": "description of what this node does",
      "model_name": "model to use (if LLM_CALL)",
      "prompt_template": "prompt template with {{variables}} (if LLM_CALL)",
      "tool_name": "tool name (if TOOL_USE)",
      "tool_params": {{"param": "value"}} (if TOOL_USE),
      "condition": "logical condition (if DECISION/CONDITION)",
      "branches": ["node_id1", "node_id2"] (if DECISION),
      "inputs": {{"key": "description"}},
      "outputs": {{"key": "description"}},
      "config": {{}}
    }}
  ],
  "edges": [
    {{
      "from": "source_node_id",
      "to": "target_node_id", 
      "data_key": "key of data being passed",
      "condition": "optional condition for edge traversal"
    }}
  ],
  "start_node": "id of starting node",
  "end_nodes": ["id of final node(s)"],
  "metadata": {{
    "complexity": "simple|medium|complex",
    "estimated_time": "estimated execution time",
    "resource_requirements": ["list of required resources"]
  }}
}}

Requirements:
1. Include proper INPUT and OUTPUT nodes
2. Ensure all nodes are connected via edges
3. Use descriptive names and operations
4. Include appropriate prompt templates for LLM_CALL nodes
5. Specify tool names and parameters for TOOL_USE nodes
6. Add conditional logic where appropriate
7. Ensure the workflow is executable and makes logical sense

Generate the JSON specification now:
"""

    WORKFLOW_REFINEMENT_PROMPT = """
Review and refine the following workflow specification based on the feedback provided.

Original Task: {task_description}
Current Workflow: {current_workflow}
Feedback/Issues: {feedback}

Please address the following aspects:
1. Fix any structural issues (missing connections, invalid node types, etc.)
2. Improve prompt templates for better LLM performance
3. Add error handling where appropriate
4. Optimize the workflow for efficiency
5. Ensure all required functionality is covered

Provide the refined workflow specification in the same JSON format as before, incorporating all improvements.

Refined JSON specification:
"""

    WORKFLOW_OPTIMIZATION_PROMPT = """
Optimize the following workflow for better performance and reliability.

Workflow: {workflow_specification}
Current Performance Issues: {performance_issues}
Optimization Goals: {optimization_goals}

Consider these optimization strategies:
1. Parallel processing opportunities
2. Caching frequently used results
3. Combining similar operations
4. Reducing redundant LLM calls
5. Improving prompt efficiency
6. Better error handling and recovery

Provide an optimized version of the workflow with explanations for each optimization made.

Optimizations Applied:
[List each optimization and why it was made]

Optimized Workflow:
[Provide the optimized JSON specification]
"""

    NODE_GENERATION_PROMPT = """
Generate a specific workflow node based on the requirements.

Node Requirements:
- Purpose: {node_purpose}
- Input Data: {input_data}
- Expected Output: {expected_output}
- Node Type Preference: {preferred_node_type}
- Context: {workflow_context}

Generate a node specification in this format:
{{
  "id": "descriptive_node_id",
  "name": "Human Readable Node Name",
  "node_type": "appropriate_node_type",
  "operation": "clear description of operation",
  "model_name": "model_if_llm_call",
  "prompt_template": "optimized prompt template with {{variables}}",
  "tool_name": "tool_if_tool_use",
  "tool_params": {{}},
  "condition": "condition_if_decision",
  "inputs": {{"key": "description"}},
  "outputs": {{"key": "description"}},
  "config": {{}}
}}

Node specification:
"""

    EDGE_GENERATION_PROMPT = """
Generate appropriate edges to connect the following nodes in a workflow.

Nodes to Connect:
{nodes_info}

Workflow Purpose: {workflow_purpose}
Data Flow Requirements: {data_flow_requirements}

For each connection, consider:
1. What data needs to flow between nodes?
2. Are there conditional connections?
3. What is the logical execution order?
4. Are there any parallel execution opportunities?

Generate edge specifications in this format:
[
  {{
    "from": "source_node_id",
    "to": "target_node_id",
    "data_key": "data_being_passed",
    "condition": "optional_condition"
  }}
]

Edge specifications:
"""

    @classmethod
    def get_workflow_analysis_prompt(
        cls, 
        task_description: str,
        complexity_level: str = "medium",
        available_tools: List[str] = None,
        model_constraints: Dict[str, Any] = None
    ) -> str:
        """Get the workflow analysis prompt with substituted values"""
        return cls.WORKFLOW_ANALYSIS_PROMPT.format(
            task_description=task_description,
            complexity_level=complexity_level,
            available_tools=available_tools or ["web_search", "calculator", "code_executor"],
            model_constraints=model_constraints or {"budget": "medium", "latency": "normal"}
        )
    
    @classmethod
    def get_workflow_generation_prompt(
        cls,
        task_description: str,
        workflow_analysis: str
    ) -> str:
        """Get the workflow generation prompt with analysis"""
        return cls.WORKFLOW_GENERATION_PROMPT.format(
            task_description=task_description,
            workflow_analysis=workflow_analysis
        )
    
    @classmethod
    def get_workflow_refinement_prompt(
        cls,
        task_description: str,
        current_workflow: str,
        feedback: str
    ) -> str:
        """Get the workflow refinement prompt"""
        return cls.WORKFLOW_REFINEMENT_PROMPT.format(
            task_description=task_description,
            current_workflow=current_workflow,
            feedback=feedback
        )
    
    @classmethod
    def get_workflow_optimization_prompt(
        cls,
        workflow_specification: str,
        performance_issues: List[str] = None,
        optimization_goals: List[str] = None
    ) -> str:
        """Get the workflow optimization prompt"""
        return cls.WORKFLOW_OPTIMIZATION_PROMPT.format(
            workflow_specification=workflow_specification,
            performance_issues=performance_issues or ["slow execution", "high cost"],
            optimization_goals=optimization_goals or ["reduce latency", "improve accuracy"]
        )
    
    @classmethod
    def get_node_generation_prompt(
        cls,
        node_purpose: str,
        input_data: str,
        expected_output: str,
        preferred_node_type: str = "LLM_CALL",
        workflow_context: str = ""
    ) -> str:
        """Get the node generation prompt"""
        return cls.NODE_GENERATION_PROMPT.format(
            node_purpose=node_purpose,
            input_data=input_data,
            expected_output=expected_output,
            preferred_node_type=preferred_node_type,
            workflow_context=workflow_context
        )
    
    @classmethod
    def get_edge_generation_prompt(
        cls,
        nodes_info: List[Dict[str, Any]],
        workflow_purpose: str,
        data_flow_requirements: str
    ) -> str:
        """Get the edge generation prompt"""
        return cls.EDGE_GENERATION_PROMPT.format(
            nodes_info=nodes_info,
            workflow_purpose=workflow_purpose,
            data_flow_requirements=data_flow_requirements
        )