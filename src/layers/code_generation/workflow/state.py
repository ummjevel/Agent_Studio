"""
Workflow state management for tracking execution state
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class NodeExecutionResult(BaseModel):
    """Result of executing a single node"""
    node_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class WorkflowState(BaseModel):
    """
    Tracks the execution state of a workflow.
    
    This maintains the current state during workflow execution,
    including which nodes have been executed, their results,
    and the overall workflow status.
    """
    
    workflow_id: str = Field(..., description="ID of the workflow being executed")
    execution_id: str = Field(..., description="Unique ID for this execution")
    
    # Overall execution state
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Node execution tracking
    current_node: Optional[str] = Field(None, description="Currently executing node")
    completed_nodes: List[str] = Field(default_factory=list, description="Nodes that have completed")
    failed_nodes: List[str] = Field(default_factory=list, description="Nodes that failed")
    node_results: Dict[str, NodeExecutionResult] = Field(default_factory=dict)
    
    # Data flow state
    shared_state: Dict[str, Any] = Field(default_factory=dict, description="Shared data between nodes")
    node_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Outputs from each node")
    
    # Error handling
    error_message: Optional[str] = None
    error_node: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def start_execution(self) -> None:
        """Mark the workflow execution as started"""
        self.status = ExecutionStatus.RUNNING
        self.start_time = datetime.now()
    
    def complete_execution(self) -> None:
        """Mark the workflow execution as completed"""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = datetime.now()
        self.current_node = None
    
    def fail_execution(self, error_message: str, error_node: Optional[str] = None) -> None:
        """Mark the workflow execution as failed"""
        self.status = ExecutionStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        self.error_node = error_node
        self.current_node = None
    
    def start_node_execution(self, node_id: str, inputs: Dict[str, Any]) -> None:
        """Mark a node as starting execution"""
        self.current_node = node_id
        
        result = NodeExecutionResult(
            node_id=node_id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now(),
            inputs=inputs
        )
        self.node_results[node_id] = result
    
    def complete_node_execution(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Mark a node as completed"""
        if node_id not in self.node_results:
            raise ValueError(f"Node {node_id} was not started")
        
        result = self.node_results[node_id]
        result.status = ExecutionStatus.COMPLETED
        result.end_time = datetime.now()
        result.outputs = outputs
        
        # Calculate execution time
        if result.start_time and result.end_time:
            execution_time = (result.end_time - result.start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
        
        # Update tracking lists
        if node_id not in self.completed_nodes:
            self.completed_nodes.append(node_id)
        
        # Store node outputs for data flow
        self.node_outputs[node_id] = outputs
        
        # Update shared state with outputs
        self.shared_state.update(outputs)
        
        self.current_node = None
    
    def fail_node_execution(self, node_id: str, error_message: str) -> None:
        """Mark a node as failed"""
        if node_id not in self.node_results:
            raise ValueError(f"Node {node_id} was not started")
        
        result = self.node_results[node_id]
        result.status = ExecutionStatus.FAILED
        result.end_time = datetime.now()
        result.error_message = error_message
        
        # Calculate execution time
        if result.start_time and result.end_time:
            execution_time = (result.end_time - result.start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
        
        # Update tracking lists
        if node_id not in self.failed_nodes:
            self.failed_nodes.append(node_id)
        
        self.current_node = None
    
    def get_node_output(self, node_id: str, key: Optional[str] = None) -> Any:
        """Get output from a specific node"""
        if node_id not in self.node_outputs:
            return None
        
        if key is None:
            return self.node_outputs[node_id]
        
        return self.node_outputs[node_id].get(key)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set data in shared state"""
        self.shared_state[key] = value
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get data from shared state"""
        return self.shared_state.get(key, default)
    
    def is_node_completed(self, node_id: str) -> bool:
        """Check if a node has completed successfully"""
        return node_id in self.completed_nodes
    
    def is_node_failed(self, node_id: str) -> bool:
        """Check if a node has failed"""
        return node_id in self.failed_nodes
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution"""
        total_execution_time = None
        if self.start_time and self.end_time:
            total_execution_time = (self.end_time - self.start_time).total_seconds() * 1000
        
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "total_nodes": len(self.node_results),
            "completed_nodes": len(self.completed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "total_execution_time_ms": total_execution_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_node": self.current_node,
            "error_message": self.error_message,
            "error_node": self.error_node
        }