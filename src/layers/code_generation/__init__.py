"""
Code Generation Layer for Self-Evolving Agent Framework

This layer implements workflow generation using a hybrid approach:
- Template-based generation for common patterns
- LLM-direct generation for complex cases
- LangGraph integration for execution
"""

from .generator import WorkflowCodeGenerator, GenerationMode
from .workflow.node import WorkflowNode, NodeType
from .workflow.graph import WorkflowGraph
from .hybrid_generator import HybridWorkflowGenerator, GenerationStrategy
from .templates.template_generator import TemplateBasedGenerator
from .llm_generators.llm_generator import LLMDirectGenerator
from .llm_client.client_factory import LLMClientFactory

__all__ = [
    "WorkflowCodeGenerator",
    "GenerationMode", 
    "WorkflowNode",
    "NodeType", 
    "WorkflowGraph",
    "HybridWorkflowGenerator",
    "GenerationStrategy",
    "TemplateBasedGenerator",
    "LLMDirectGenerator",
    "LLMClientFactory"
]