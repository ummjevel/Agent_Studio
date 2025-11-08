"""
WorkflowNode implementation for representing individual workflow steps
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class NodeType(str, Enum):
    """Types of workflow nodes"""
    LLM_CALL = "llm_call"
    TOOL_USE = "tool_use"
    DECISION = "decision"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    INPUT = "input"
    OUTPUT = "output"
    TRANSFORM = "transform"
    CONDITION = "condition"


class WorkflowNode(BaseModel):
    """
    Represents a single node in a workflow graph.
    
    Based on the design from introduce.md:
    - 노드: LLM 호출, 도구 사용, 제어 흐름
    - 엣지: 데이터 흐름 및 의존성
    """
    
    id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Human-readable name for the node")
    node_type: NodeType = Field(..., description="Type of the node")
    operation: str = Field(..., description="The operation this node performs")
    
    # Input/Output configuration
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input schema/data for the node")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output schema/data from the node")
    
    # Node-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Node-specific configuration")
    
    # LLM-specific fields
    model_name: Optional[str] = Field(None, description="LLM model to use for LLM_CALL nodes")
    prompt_template: Optional[str] = Field(None, description="Prompt template for LLM_CALL nodes")
    
    # Tool-specific fields
    tool_name: Optional[str] = Field(None, description="Tool name for TOOL_USE nodes")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    
    # Decision/Condition fields
    condition: Optional[str] = Field(None, description="Condition logic for DECISION/CONDITION nodes")
    branches: List[str] = Field(default_factory=list, description="Branch node IDs")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node operation.
        This is a placeholder - actual execution will be handled by the workflow engine.
        """
        # Store input data
        self.inputs.update(input_data)
        
        # Placeholder execution logic
        if self.node_type == NodeType.LLM_CALL:
            return self._execute_llm_call(input_data)
        elif self.node_type == NodeType.TOOL_USE:
            return self._execute_tool_use(input_data)
        elif self.node_type == NodeType.DECISION:
            return self._execute_decision(input_data)
        elif self.node_type == NodeType.TRANSFORM:
            return self._execute_transform(input_data)
        else:
            # Default pass-through
            return input_data
    
    def _execute_llm_call(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM call node"""
        # This will be implemented with actual LLM integration
        return {"result": f"LLM response for {self.operation}", "node_id": self.id}
    
    def _execute_tool_use(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool use node"""
        # This will be implemented with actual tool integration
        return {"result": f"Tool {self.tool_name} executed", "node_id": self.id}
    
    def _execute_decision(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision node"""
        # Simple condition evaluation placeholder
        return {"branch": self.branches[0] if self.branches else None, "node_id": self.id}
    
    def _execute_transform(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformation node"""
        # Simple transformation placeholder
        transformed = input_data.copy()
        transformed["transformed_by"] = self.id
        return transformed
    
    def validate_configuration(self) -> bool:
        """Validate node configuration based on node type"""
        if self.node_type == NodeType.LLM_CALL:
            return self.model_name is not None and self.prompt_template is not None
        elif self.node_type == NodeType.TOOL_USE:
            return self.tool_name is not None
        elif self.node_type in [NodeType.DECISION, NodeType.CONDITION]:
            return self.condition is not None
        return True
    
    def to_langgraph_node(self) -> Dict[str, Any]:
        """Convert to LangGraph node format"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "config": {
                "operation": self.operation,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "node_config": self.config,
                **self._get_type_specific_config()
            }
        }
    
    def _get_type_specific_config(self) -> Dict[str, Any]:
        """Get type-specific configuration for LangGraph conversion"""
        config = {}
        
        if self.node_type == NodeType.LLM_CALL:
            config.update({
                "model_name": self.model_name,
                "prompt_template": self.prompt_template
            })
        elif self.node_type == NodeType.TOOL_USE:
            config.update({
                "tool_name": self.tool_name,
                "tool_params": self.tool_params
            })
        elif self.node_type in [NodeType.DECISION, NodeType.CONDITION]:
            config.update({
                "condition": self.condition,
                "branches": self.branches
            })
        
        return config