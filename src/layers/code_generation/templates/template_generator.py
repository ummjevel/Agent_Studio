"""
Template-based workflow generator implementation
"""

from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

from ..workflow.node import WorkflowNode, NodeType
from ..workflow.graph import WorkflowGraph
from .workflow_templates import WorkflowTemplateLibrary, WorkflowTemplate, TaskType


class TemplateBasedGenerator:
    """
    Generates workflows based on pre-defined templates.
    
    This provides a reliable, fast way to create workflows for common patterns
    by using parameterizable templates.
    """
    
    def __init__(self):
        self.template_library = WorkflowTemplateLibrary()
    
    def generate_workflow(
        self,
        template_id: str,
        parameters: Dict[str, Any],
        workflow_name: Optional[str] = None,
        workflow_description: Optional[str] = None
    ) -> WorkflowGraph:
        """
        Generate a workflow from a template.
        
        Args:
            template_id: ID of the template to use
            parameters: Parameters to substitute in the template
            workflow_name: Optional custom name for the workflow
            workflow_description: Optional custom description
            
        Returns:
            Generated WorkflowGraph instance
            
        Raises:
            ValueError: If template not found or required parameters missing
        """
        # Get the template
        template = self.template_library.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Validate required parameters
        missing_params = [param for param in template.required_params 
                         if param not in parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Generate workflow ID and metadata
        workflow_id = str(uuid.uuid4())
        name = workflow_name or f"{template.name} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = workflow_description or f"Generated from template: {template.name}"
        
        # Create workflow
        workflow = WorkflowGraph(
            id=workflow_id,
            name=name,
            description=description,
            tags=template.tags + ["template_generated"],
            metadata={
                "template_id": template_id,
                "template_name": template.name,
                "generated_at": datetime.now().isoformat(),
                "parameters_used": parameters
            }
        )
        
        # Generate nodes from template
        for node_template in template.node_templates:
            node = self._create_node_from_template(node_template, parameters)
            workflow.add_node(node)
        
        # Generate edges from template
        for edge_template in template.edge_templates:
            self._create_edge_from_template(workflow, edge_template, parameters)
        
        return workflow
    
    def _create_node_from_template(
        self, 
        node_template: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> WorkflowNode:
        """Create a WorkflowNode from a node template"""
        
        # Substitute parameters in template values
        substituted_template = self._substitute_parameters(node_template, parameters)
        
        # Create node
        node = WorkflowNode(
            id=substituted_template["id"],
            name=substituted_template["name"],
            node_type=NodeType(substituted_template["node_type"]),
            operation=substituted_template["operation"]
        )
        
        # Set type-specific fields
        if "model_name" in substituted_template:
            node.model_name = substituted_template["model_name"]
        
        if "prompt_template" in substituted_template:
            node.prompt_template = substituted_template["prompt_template"]
        
        if "tool_name" in substituted_template:
            node.tool_name = substituted_template["tool_name"]
        
        if "tool_params" in substituted_template:
            node.tool_params = substituted_template["tool_params"]
        
        if "condition" in substituted_template:
            node.condition = substituted_template["condition"]
        
        if "branches" in substituted_template:
            node.branches = substituted_template["branches"]
        
        # Set configuration
        if "config" in substituted_template:
            node.config = substituted_template["config"]
        
        return node
    
    def _create_edge_from_template(
        self,
        workflow: WorkflowGraph,
        edge_template: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> None:
        """Create an edge from an edge template"""
        
        # Substitute parameters
        substituted_template = self._substitute_parameters(edge_template, parameters)
        
        # Create edge
        workflow.connect(
            from_node_id=substituted_template["from"],
            to_node_id=substituted_template["to"],
            data_key=substituted_template["data_key"],
            condition=substituted_template.get("condition")
        )
    
    def _substitute_parameters(
        self, 
        template_dict: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Substitute parameters in a template dictionary"""
        
        def substitute_value(value):
            if isinstance(value, str):
                # Simple parameter substitution using format strings
                try:
                    return value.format(**parameters)
                except KeyError as e:
                    # If parameter is missing, leave placeholder as is
                    return value
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return {key: substitute_value(value) for key, value in template_dict.items()}
    
    def generate_workflow_auto(
        self,
        task_description: str,
        complexity_hint: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> WorkflowGraph:
        """
        Automatically generate a workflow based on task description.
        
        Args:
            task_description: Description of the task
            complexity_hint: Hint about complexity level ("simple", "medium", "complex")
            additional_params: Additional parameters for the template
            
        Returns:
            Generated WorkflowGraph instance
        """
        # Recommend a template
        recommended_template = self.template_library.recommend_template(
            task_description, complexity_hint
        )
        
        if not recommended_template:
            # Fallback to simple Q&A
            recommended_template = self.template_library.get_template("simple_qa")
        
        # Prepare default parameters
        default_params = {
            "model_name": "gpt-4",  # Default model
            "task_description": task_description
        }
        
        if additional_params:
            default_params.update(additional_params)
        
        # Generate workflow
        return self.generate_workflow(
            template_id=recommended_template.id,
            parameters=default_params,
            workflow_name=f"Auto-generated: {task_description[:50]}...",
            workflow_description=f"Auto-generated workflow for: {task_description}"
        )
    
    def customize_template(
        self,
        base_template_id: str,
        customizations: Dict[str, Any],
        new_template_id: str,
        new_template_name: str
    ) -> WorkflowTemplate:
        """
        Create a customized version of an existing template.
        
        Args:
            base_template_id: ID of the base template
            customizations: Dictionary of customizations to apply
            new_template_id: ID for the new template
            new_template_name: Name for the new template
            
        Returns:
            New customized WorkflowTemplate
        """
        base_template = self.template_library.get_template(base_template_id)
        if not base_template:
            raise ValueError(f"Base template '{base_template_id}' not found")
        
        # Create a copy of the base template
        custom_template = WorkflowTemplate(
            id=new_template_id,
            name=new_template_name,
            description=f"Customized from {base_template.name}",
            task_type=base_template.task_type,
            complexity_level=base_template.complexity_level,
            node_templates=base_template.node_templates.copy(),
            edge_templates=base_template.edge_templates.copy(),
            parameters=base_template.parameters.copy(),
            required_params=base_template.required_params.copy(),
            tags=base_template.tags + ["customized"],
            use_cases=base_template.use_cases.copy()
        )
        
        # Apply customizations
        if "node_templates" in customizations:
            custom_template.node_templates = customizations["node_templates"]
        
        if "edge_templates" in customizations:
            custom_template.edge_templates = customizations["edge_templates"]
        
        if "required_params" in customizations:
            custom_template.required_params = customizations["required_params"]
        
        if "complexity_level" in customizations:
            custom_template.complexity_level = customizations["complexity_level"]
        
        # Add to library
        self.template_library.add_template(custom_template)
        
        return custom_template
    
    def validate_template_parameters(
        self, 
        template_id: str, 
        parameters: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate parameters for a template.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        template = self.template_library.get_template(template_id)
        if not template:
            return False, [f"Template '{template_id}' not found"]
        
        errors = []
        
        # Check required parameters
        for required_param in template.required_params:
            if required_param not in parameters:
                errors.append(f"Missing required parameter: {required_param}")
        
        # Additional validation could be added here
        # (e.g., parameter type checking, value range validation)
        
        return len(errors) == 0, errors
    
    def get_template_info(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template"""
        template = self.template_library.get_template(template_id)
        if not template:
            return None
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "task_type": template.task_type.value,
            "complexity_level": template.complexity_level,
            "required_params": template.required_params,
            "tags": template.tags,
            "use_cases": template.use_cases,
            "node_count": len(template.node_templates),
            "edge_count": len(template.edge_templates)
        }
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with basic info"""
        templates_info = []
        for template in self.template_library.list_all_templates():
            templates_info.append({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "task_type": template.task_type.value,
                "complexity_level": template.complexity_level,
                "tags": template.tags
            })
        return templates_info
    
    def export_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Export a template as a dictionary for sharing or storage"""
        template = self.template_library.get_template(template_id)
        if not template:
            return None
        
        return template.model_dump()
    
    def import_template(self, template_data: Dict[str, Any]) -> WorkflowTemplate:
        """Import a template from dictionary data"""
        template = WorkflowTemplate(**template_data)
        self.template_library.add_template(template)
        return template