"""
WorkflowGraph implementation for managing workflow structure
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json

from .node import WorkflowNode, NodeType


class WorkflowEdge(BaseModel):
    """Represents an edge between workflow nodes"""
    
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    data_key: str = Field(..., description="Data key being passed")
    condition: Optional[str] = Field(None, description="Condition for edge traversal")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowGraph(BaseModel):
    """
    Represents a complete workflow as a directed graph.
    
    Based on introduce.md design:
    - 노드: LLM 호출, 도구 사용, 제어 흐름
    - 엣지: 데이터 흐름 및 의존성
    """
    
    id: str = Field(..., description="Unique identifier for the workflow")
    name: str = Field(..., description="Human-readable name for the workflow")
    description: str = Field("", description="Description of the workflow")
    
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict, description="Nodes in the workflow")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="Edges connecting nodes")
    
    # Entry and exit points
    start_node: Optional[str] = Field(None, description="Starting node ID")
    end_nodes: List[str] = Field(default_factory=list, description="Ending node IDs")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field("1.0", description="Workflow version")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow"""
        self.nodes[node.id] = node
        
        # Auto-set start node if it's the first INPUT node
        if node.node_type == NodeType.INPUT and self.start_node is None:
            self.start_node = node.id
        
        # Auto-add to end nodes if it's an OUTPUT node
        if node.node_type == NodeType.OUTPUT and node.id not in self.end_nodes:
            self.end_nodes.append(node.id)
    
    def connect(self, from_node_id: str, to_node_id: str, data_key: str, condition: Optional[str] = None) -> None:
        """Connect two nodes with an edge"""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found in workflow")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found in workflow")
        
        edge = WorkflowEdge(
            from_node=from_node_id,
            to_node=to_node_id,
            data_key=data_key,
            condition=condition
        )
        self.edges.append(edge)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all connected edges"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in workflow")
        
        # Remove the node
        del self.nodes[node_id]
        
        # Remove all edges connected to this node
        self.edges = [edge for edge in self.edges 
                     if edge.from_node != node_id and edge.to_node != node_id]
        
        # Update start/end nodes if necessary
        if self.start_node == node_id:
            self.start_node = None
        if node_id in self.end_nodes:
            self.end_nodes.remove(node_id)
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all nodes that directly feed into the given node"""
        return [edge.from_node for edge in self.edges if edge.to_node == node_id]
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get all nodes that the given node directly feeds into"""
        return [edge.to_node for edge in self.edges if edge.from_node == node_id]
    
    def get_execution_order(self) -> List[str]:
        """Get topological order for workflow execution"""
        # Simple topological sort implementation
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        # Calculate in-degrees
        for edge in self.edges:
            in_degree[edge.to_node] += 1
        
        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for successor nodes
            for successor in self.get_successors(current):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Check for cycles
        if len(result) != len(self.nodes):
            raise ValueError("Workflow contains cycles - cannot determine execution order")
        
        return result
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the workflow structure"""
        errors = []
        
        # Check if we have nodes
        if not self.nodes:
            errors.append("Workflow has no nodes")
        
        # Check for start node
        if self.start_node is None:
            errors.append("No start node specified")
        elif self.start_node not in self.nodes:
            errors.append(f"Start node {self.start_node} not found in workflow")
        
        # Check for end nodes
        if not self.end_nodes:
            errors.append("No end nodes specified")
        
        # Validate each node
        for node_id, node in self.nodes.items():
            if not node.validate_configuration():
                errors.append(f"Node {node_id} has invalid configuration")
        
        # Check for orphaned nodes (except start node)
        for node_id in self.nodes:
            if node_id != self.start_node:
                predecessors = self.get_predecessors(node_id)
                if not predecessors:
                    errors.append(f"Node {node_id} has no incoming connections")
        
        # Check for unreachable end nodes
        for end_node in self.end_nodes:
            if end_node not in self.nodes:
                errors.append(f"End node {end_node} not found in workflow")
        
        # Try to get execution order (will catch cycles)
        try:
            self.get_execution_order()
        except ValueError as e:
            errors.append(str(e))
        
        return len(errors) == 0, errors
    
    def to_langgraph(self) -> Dict[str, Any]:
        """Convert workflow to LangGraph format"""
        langgraph_nodes = {}
        langgraph_edges = []
        
        # Convert nodes
        for node_id, node in self.nodes.items():
            langgraph_nodes[node_id] = node.to_langgraph_node()
        
        # Convert edges
        for edge in self.edges:
            langgraph_edges.append({
                "from": edge.from_node,
                "to": edge.to_node,
                "data_key": edge.data_key,
                "condition": edge.condition
            })
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": langgraph_nodes,
            "edges": langgraph_edges,
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "metadata": self.metadata
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        node_types = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "has_cycles": False,  # We'd detect this in validation
            "complexity_score": len(self.nodes) + len(self.edges),
            "start_node": self.start_node,
            "end_nodes_count": len(self.end_nodes)
        }
    
    def clone(self, new_id: str, new_name: str) -> 'WorkflowGraph':
        """Create a copy of the workflow with new ID and name"""
        cloned_nodes = {node_id: WorkflowNode(**node.dict()) for node_id, node in self.nodes.items()}
        cloned_edges = [WorkflowEdge(**edge.dict()) for edge in self.edges]
        
        return WorkflowGraph(
            id=new_id,
            name=new_name,
            description=f"Cloned from {self.name}",
            nodes=cloned_nodes,
            edges=cloned_edges,
            start_node=self.start_node,
            end_nodes=self.end_nodes.copy(),
            tags=self.tags.copy(),
            metadata=self.metadata.copy()
        )