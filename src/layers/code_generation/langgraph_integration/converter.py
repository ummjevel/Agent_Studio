"""
Converts WorkflowGraph to LangGraph format for execution
"""

from typing import Any, Dict, List, Optional, Callable
from ..workflow.graph import WorkflowGraph
from ..workflow.node import WorkflowNode, NodeType
from ..workflow.state import WorkflowState


class WorkflowToLangGraphConverter:
    """
    Converts our WorkflowGraph representation to LangGraph format.
    
    This enables execution of workflows using the LangGraph framework
    while maintaining our own workflow abstraction.
    """
    
    def __init__(self):
        self.node_function_map: Dict[str, Callable] = {}
    
    def convert(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """
        Convert a WorkflowGraph to LangGraph configuration.
        
        Returns a dictionary that can be used to create a LangGraph StateGraph.
        """
        # Validate workflow first
        is_valid, errors = workflow.validate()
        if not is_valid:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")
        
        # Convert to LangGraph format
        langgraph_config = {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "nodes": self._convert_nodes(workflow),
            "edges": self._convert_edges(workflow),
            "start_node": workflow.start_node,
            "end_nodes": workflow.end_nodes,
            "state_schema": self._generate_state_schema(workflow),
            "node_functions": self._generate_node_functions(workflow)
        }
        
        return langgraph_config
    
    def _convert_nodes(self, workflow: WorkflowGraph) -> Dict[str, Dict[str, Any]]:
        """Convert workflow nodes to LangGraph node format"""
        langgraph_nodes = {}
        
        for node_id, node in workflow.nodes.items():
            langgraph_nodes[node_id] = {
                "id": node_id,
                "name": node.name,
                "type": node.node_type.value,
                "function_name": f"execute_{node_id}",
                "config": self._get_node_config(node)
            }
        
        return langgraph_nodes
    
    def _convert_edges(self, workflow: WorkflowGraph) -> List[Dict[str, Any]]:
        """Convert workflow edges to LangGraph edge format"""
        langgraph_edges = []
        
        for edge in workflow.edges:
            edge_config = {
                "from": edge.from_node,
                "to": edge.to_node,
                "data_key": edge.data_key
            }
            
            # Add condition if present
            if edge.condition:
                edge_config["condition"] = edge.condition
            
            langgraph_edges.append(edge_config)
        
        return langgraph_edges
    
    def _get_node_config(self, node: WorkflowNode) -> Dict[str, Any]:
        """Get LangGraph-specific configuration for a node"""
        config = {
            "operation": node.operation,
            "inputs": node.inputs,
            "outputs": node.outputs,
            "metadata": node.metadata
        }
        
        # Add type-specific configuration
        if node.node_type == NodeType.LLM_CALL:
            config.update({
                "model_name": node.model_name,
                "prompt_template": node.prompt_template
            })
        elif node.node_type == NodeType.TOOL_USE:
            config.update({
                "tool_name": node.tool_name,
                "tool_params": node.tool_params
            })
        elif node.node_type in [NodeType.DECISION, NodeType.CONDITION]:
            config.update({
                "condition": node.condition,
                "branches": node.branches
            })
        
        return config
    
    def _generate_state_schema(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """Generate state schema for LangGraph"""
        # Basic state schema - can be extended based on workflow requirements
        schema = {
            "workflow_id": "str",
            "execution_id": "str",
            "current_step": "str",
            "shared_data": "dict",
            "node_outputs": "dict",
            "error_message": "optional[str]",
            "metadata": "dict"
        }
        
        # Add input/output schemas from nodes
        for node_id, node in workflow.nodes.items():
            if node.inputs:
                schema[f"{node_id}_inputs"] = "dict"
            if node.outputs:
                schema[f"{node_id}_outputs"] = "dict"
        
        return schema
    
    def _generate_node_functions(self, workflow: WorkflowGraph) -> Dict[str, Callable]:
        """Generate execution functions for each node"""
        node_functions = {}
        
        for node_id, node in workflow.nodes.items():
            function_name = f"execute_{node_id}"
            node_functions[function_name] = self._create_node_function(node)
        
        return node_functions
    
    def _create_node_function(self, node: WorkflowNode) -> Callable:
        """Create an execution function for a specific node"""
        
        def node_executor(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute a single node within LangGraph context.
            
            Args:
                state: Current workflow state
                
            Returns:
                Updated state after node execution
            """
            try:
                # Extract input data from state
                input_data = self._extract_node_inputs(node, state)
                
                # Execute the node
                output_data = node.execute(input_data)
                
                # Update state with outputs
                updated_state = state.copy()
                updated_state["current_step"] = node.id
                updated_state["node_outputs"] = updated_state.get("node_outputs", {})
                updated_state["node_outputs"][node.id] = output_data
                
                # Merge outputs into shared data
                shared_data = updated_state.get("shared_data", {})
                shared_data.update(output_data)
                updated_state["shared_data"] = shared_data
                
                return updated_state
                
            except Exception as e:
                # Handle node execution errors
                error_state = state.copy()
                error_state["error_message"] = f"Error in node {node.id}: {str(e)}"
                error_state["current_step"] = node.id
                return error_state
        
        # Set function metadata
        node_executor.__name__ = f"execute_{node.id}"
        node_executor.__doc__ = f"Execute workflow node: {node.name}"
        
        return node_executor
    
    def _extract_node_inputs(self, node: WorkflowNode, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input data for a node from the workflow state"""
        input_data = {}
        
        # Get shared data
        shared_data = state.get("shared_data", {})
        
        # Get outputs from previous nodes if specified in node inputs
        node_outputs = state.get("node_outputs", {})
        
        # Start with shared data
        input_data.update(shared_data)
        
        # Add specific inputs if defined
        if node.inputs:
            for key, value in node.inputs.items():
                if isinstance(value, str) and value.startswith("node:"):
                    # Reference to another node's output
                    source_node = value.split(":")[1]
                    if source_node in node_outputs:
                        input_data[key] = node_outputs[source_node]
                else:
                    # Direct value
                    input_data[key] = value
        
        return input_data
    
    def generate_langgraph_code(self, workflow: WorkflowGraph) -> str:
        """
        Generate Python code that creates a LangGraph StateGraph.
        
        This is useful for debugging or when you need the actual LangGraph code.
        """
        config = self.convert(workflow)
        
        code_lines = [
            "from langgraph.graph import StateGraph",
            "from typing import Dict, Any",
            "",
            "# Generated LangGraph code",
            f"# Workflow: {workflow.name}",
            f"# ID: {workflow.id}",
            "",
            "def create_workflow_graph():",
            "    \"\"\"Create and configure the LangGraph StateGraph\"\"\"",
            "",
            "    # Define state schema",
            "    class WorkflowState:",
            "        pass  # Add proper TypedDict here",
            "",
            "    # Create graph",
            "    graph = StateGraph(WorkflowState)",
            ""
        ]
        
        # Add node functions
        for node_id, node in workflow.nodes.items():
            code_lines.extend([
                f"    def execute_{node_id}(state: Dict[str, Any]) -> Dict[str, Any]:",
                f"        \"\"\"Execute node: {node.name}\"\"\"",
                f"        # Node type: {node.node_type.value}",
                f"        # Operation: {node.operation}",
                "        # TODO: Implement actual node execution logic",
                "        return state",
                ""
            ])
        
        # Add nodes to graph
        code_lines.append("    # Add nodes to graph")
        for node_id in workflow.nodes:
            code_lines.append(f"    graph.add_node('{node_id}', execute_{node_id})")
        
        code_lines.append("")
        
        # Add edges
        code_lines.append("    # Add edges")
        for edge in workflow.edges:
            if edge.condition:
                code_lines.append(
                    f"    graph.add_conditional_edge('{edge.from_node}', "
                    f"lambda state: '{edge.to_node}' if {edge.condition} else None)"
                )
            else:
                code_lines.append(f"    graph.add_edge('{edge.from_node}', '{edge.to_node}')")
        
        # Set entry point
        if workflow.start_node:
            code_lines.extend([
                "",
                f"    # Set entry point",
                f"    graph.set_entry_point('{workflow.start_node}')",
                "",
                "    return graph.compile()"
            ])
        
        return "\n".join(code_lines)