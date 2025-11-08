"""
Executes workflows using LangGraph framework
"""

from typing import Any, Dict, Optional, Iterator, AsyncIterator
from datetime import datetime
import asyncio
import uuid

from ..workflow.graph import WorkflowGraph
from ..workflow.state import WorkflowState, ExecutionStatus
from .converter import WorkflowToLangGraphConverter


class LangGraphExecutor:
    """
    Executes workflows using LangGraph framework.
    
    This class provides both synchronous and asynchronous execution
    of workflows that have been converted to LangGraph format.
    """
    
    def __init__(self):
        self.converter = WorkflowToLangGraphConverter()
        self.active_executions: Dict[str, WorkflowState] = {}
    
    def execute_workflow(
        self, 
        workflow: WorkflowGraph, 
        initial_data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Execute a workflow synchronously.
        
        Args:
            workflow: The workflow to execute
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID (generated if not provided)
            
        Returns:
            WorkflowState with execution results
        """
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow.id,
            execution_id=execution_id
        )
        
        # Add to active executions
        self.active_executions[execution_id] = workflow_state
        
        try:
            # Convert workflow to LangGraph format
            langgraph_config = self.converter.convert(workflow)
            
            # Initialize execution
            workflow_state.start_execution()
            
            # Set initial data
            if initial_data:
                workflow_state.shared_state.update(initial_data)
            
            # Execute workflow step by step
            execution_order = workflow.get_execution_order()
            
            for node_id in execution_order:
                if workflow_state.status == ExecutionStatus.FAILED:
                    break
                
                node = workflow.nodes[node_id]
                
                try:
                    # Prepare node inputs
                    node_inputs = self._prepare_node_inputs(node_id, workflow_state, workflow)
                    
                    # Start node execution
                    workflow_state.start_node_execution(node_id, node_inputs)
                    
                    # Execute node
                    node_outputs = node.execute(node_inputs)
                    
                    # Complete node execution
                    workflow_state.complete_node_execution(node_id, node_outputs)
                    
                except Exception as e:
                    # Handle node failure
                    error_message = f"Node {node_id} failed: {str(e)}"
                    workflow_state.fail_node_execution(node_id, error_message)
                    workflow_state.fail_execution(error_message, node_id)
                    break
            
            # Complete execution if no failures
            if workflow_state.status == ExecutionStatus.RUNNING:
                workflow_state.complete_execution()
                
        except Exception as e:
            # Handle workflow-level failure
            workflow_state.fail_execution(f"Workflow execution failed: {str(e)}")
        
        finally:
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        return workflow_state
    
    async def execute_workflow_async(
        self,
        workflow: WorkflowGraph,
        initial_data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Execute a workflow asynchronously.
        
        Args:
            workflow: The workflow to execute
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID (generated if not provided)
            
        Returns:
            WorkflowState with execution results
        """
        # For now, run the sync version in a thread pool
        # TODO: Implement proper async execution with LangGraph
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.execute_workflow, 
            workflow, 
            initial_data, 
            execution_id
        )
    
    def execute_workflow_streaming(
        self,
        workflow: WorkflowGraph,
        initial_data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> Iterator[WorkflowState]:
        """
        Execute a workflow with streaming updates.
        
        Yields workflow state after each node execution.
        
        Args:
            workflow: The workflow to execute
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID (generated if not provided)
            
        Yields:
            WorkflowState updates during execution
        """
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow.id,
            execution_id=execution_id
        )
        
        # Add to active executions
        self.active_executions[execution_id] = workflow_state
        
        try:
            # Convert workflow to LangGraph format
            langgraph_config = self.converter.convert(workflow)
            
            # Initialize execution
            workflow_state.start_execution()
            
            # Set initial data
            if initial_data:
                workflow_state.shared_state.update(initial_data)
            
            # Yield initial state
            yield workflow_state
            
            # Execute workflow step by step
            execution_order = workflow.get_execution_order()
            
            for node_id in execution_order:
                if workflow_state.status == ExecutionStatus.FAILED:
                    break
                
                node = workflow.nodes[node_id]
                
                try:
                    # Prepare node inputs
                    node_inputs = self._prepare_node_inputs(node_id, workflow_state, workflow)
                    
                    # Start node execution
                    workflow_state.start_node_execution(node_id, node_inputs)
                    yield workflow_state  # Yield state with node started
                    
                    # Execute node
                    node_outputs = node.execute(node_inputs)
                    
                    # Complete node execution
                    workflow_state.complete_node_execution(node_id, node_outputs)
                    yield workflow_state  # Yield state with node completed
                    
                except Exception as e:
                    # Handle node failure
                    error_message = f"Node {node_id} failed: {str(e)}"
                    workflow_state.fail_node_execution(node_id, error_message)
                    workflow_state.fail_execution(error_message, node_id)
                    yield workflow_state  # Yield failed state
                    break
            
            # Complete execution if no failures
            if workflow_state.status == ExecutionStatus.RUNNING:
                workflow_state.complete_execution()
                yield workflow_state  # Yield final state
                
        except Exception as e:
            # Handle workflow-level failure
            workflow_state.fail_execution(f"Workflow execution failed: {str(e)}")
            yield workflow_state  # Yield failed state
        
        finally:
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def execute_workflow_streaming_async(
        self,
        workflow: WorkflowGraph,
        initial_data: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> AsyncIterator[WorkflowState]:
        """
        Execute a workflow with async streaming updates.
        
        Args:
            workflow: The workflow to execute
            initial_data: Initial data to pass to the workflow
            execution_id: Optional execution ID (generated if not provided)
            
        Yields:
            WorkflowState updates during execution
        """
        # Convert sync streaming to async
        for state in self.execute_workflow_streaming(workflow, initial_data, execution_id):
            yield state
            await asyncio.sleep(0)  # Allow other coroutines to run
    
    def _prepare_node_inputs(
        self, 
        node_id: str, 
        workflow_state: WorkflowState, 
        workflow: WorkflowGraph
    ) -> Dict[str, Any]:
        """
        Prepare input data for a node execution.
        
        This extracts the necessary data from the workflow state
        and previous node outputs.
        """
        node = workflow.nodes[node_id]
        inputs = {}
        
        # Start with shared state
        inputs.update(workflow_state.shared_state)
        
        # Add outputs from predecessor nodes
        predecessors = workflow.get_predecessors(node_id)
        for pred_id in predecessors:
            if pred_id in workflow_state.node_outputs:
                pred_outputs = workflow_state.node_outputs[pred_id]
                inputs.update(pred_outputs)
        
        # Add node-specific inputs
        if node.inputs:
            for key, value in node.inputs.items():
                if isinstance(value, str) and value.startswith("node:"):
                    # Reference to another node's output
                    source_node = value.split(":")[1]
                    if source_node in workflow_state.node_outputs:
                        inputs[key] = workflow_state.node_outputs[source_node]
                else:
                    # Direct value
                    inputs[key] = value
        
        return inputs
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowState]:
        """Get the current status of an active execution"""
        return self.active_executions.get(execution_id)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            workflow_state = self.active_executions[execution_id]
            workflow_state.fail_execution("Execution cancelled by user")
            del self.active_executions[execution_id]
            return True
        return False
    
    def get_active_executions(self) -> Dict[str, WorkflowState]:
        """Get all currently active executions"""
        return self.active_executions.copy()
    
    def create_langgraph_instance(self, workflow: WorkflowGraph) -> Any:
        """
        Create an actual LangGraph StateGraph instance.
        
        This is for when you want to use the native LangGraph execution
        instead of our custom executor.
        
        Note: This requires LangGraph to be installed.
        """
        try:
            from langgraph.graph import StateGraph
            from typing import TypedDict
        except ImportError:
            raise ImportError("LangGraph is required for this functionality. Install with: pip install langgraph")
        
        # Convert workflow
        config = self.converter.convert(workflow)
        
        # Create state class
        class WorkflowStateDict(TypedDict):
            workflow_id: str
            execution_id: str
            current_step: str
            shared_data: dict
            node_outputs: dict
            error_message: str
        
        # Create graph
        graph = StateGraph(WorkflowStateDict)
        
        # Add nodes
        node_functions = config["node_functions"]
        for node_id in workflow.nodes:
            function_name = f"execute_{node_id}"
            if function_name in node_functions:
                graph.add_node(node_id, node_functions[function_name])
        
        # Add edges
        for edge in config["edges"]:
            if "condition" in edge:
                # Conditional edge - simplified for now
                graph.add_edge(edge["from"], edge["to"])
            else:
                graph.add_edge(edge["from"], edge["to"])
        
        # Set entry point
        if config["start_node"]:
            graph.set_entry_point(config["start_node"])
        
        return graph.compile()