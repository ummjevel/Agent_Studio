"""
Factory for creating LLM clients with integration to Layer 1 (Model Selection)
"""

from typing import Dict, Any, Optional, List
import os

from .llm_client import LLMClient, LLMProvider
from .providers import OpenAIProvider, AnthropicProvider, LiteLLMProvider, AzureOpenAIProvider, OllamaProvider


class LLMClientFactory:
    """
    Factory for creating and managing LLM clients.
    
    This integrates with Layer 1 (Model Selection) to dynamically
    choose the best LLM provider and model for each request.
    """
    
    def __init__(self):
        self.clients: Dict[LLMProvider, LLMClient] = {}
        self.model_to_provider_map: Dict[str, LLMProvider] = {}
        self._initialize_model_mappings()
    
    def _initialize_model_mappings(self):
        """Initialize mapping of models to their providers"""
        # OpenAI models
        openai_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        for model in openai_models:
            self.model_to_provider_map[model] = LLMProvider.OPENAI
        
        # Azure OpenAI models (similar to OpenAI but different provider)
        azure_models = ["gpt-35-turbo", "gpt-35-turbo-16k"]
        for model in azure_models:
            self.model_to_provider_map[model] = LLMProvider.AZURE_OPENAI
        
        # Anthropic models
        anthropic_models = [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307", "claude-2.1", "claude-2.0"
        ]
        for model in anthropic_models:
            self.model_to_provider_map[model] = LLMProvider.ANTHROPIC
        
        # Ollama models
        ollama_models = [
            "llama2", "llama2:13b", "llama2:70b",
            "codellama", "codellama:13b", "codellama:34b",
            "mistral", "mistral:7b", "mixtral:8x7b",
            "gemma:2b", "gemma:7b"
        ]
        for model in ollama_models:
            self.model_to_provider_map[model] = LLMProvider.OLLAMA
        
        # LiteLLM models (can override specific models)
        litellm_models = [
            "gemini-pro", "palm-2", "cohere/command",
            "huggingface/microsoft/DialoGPT-medium",
            "huggingface/facebook/blenderbot-400M-distill"
        ]
        for model in litellm_models:
            self.model_to_provider_map[model] = LLMProvider.LITELLM
    
    def create_client(
        self, 
        provider: LLMProvider, 
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMClient:
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: LLM provider to create client for
            api_key: API key (if None, will try to get from environment)
            **kwargs: Additional configuration for the client
            
        Returns:
            LLM client instance
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = self._get_api_key_from_env(provider)
        
        # Create client based on provider
        if provider == LLMProvider.OPENAI:
            client = OpenAIProvider(api_key=api_key, **kwargs)
        elif provider == LLMProvider.AZURE_OPENAI:
            client = AzureOpenAIProvider(api_key=api_key, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            client = AnthropicProvider(api_key=api_key, **kwargs)
        elif provider == LLMProvider.OLLAMA:
            client = OllamaProvider(api_key=api_key, **kwargs)
        elif provider == LLMProvider.LITELLM:
            client = LiteLLMProvider(api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Cache the client
        self.clients[provider] = client
        
        return client
    
    def get_client_for_model(self, model: str) -> LLMClient:
        """
        Get the appropriate client for a specific model.
        
        This integrates with Layer 1 model selection logic.
        
        Args:
            model: Model name to get client for
            
        Returns:
            LLM client that can handle the model
            
        Raises:
            ValueError: If model is not supported
        """
        # Determine provider for the model
        provider = self.model_to_provider_map.get(model)
        
        if provider is None:
            # Try to infer provider from model name
            if model.startswith(("gpt-", "text-davinci", "text-curie")):
                provider = LLMProvider.OPENAI
            elif model.startswith("gpt-35"):  # Azure naming convention
                provider = LLMProvider.AZURE_OPENAI
            elif model.startswith(("claude-", "claude_")):
                provider = LLMProvider.ANTHROPIC
            elif model in ["llama2", "codellama", "mistral", "mixtral", "gemma"] or ":" in model:
                provider = LLMProvider.OLLAMA
            else:
                # Default to LiteLLM for unknown models
                provider = LLMProvider.LITELLM
        
        # Get or create client for provider
        if provider not in self.clients:
            self.create_client(provider)
        
        client = self.clients[provider]
        
        # Validate that the client supports this model
        if not client.validate_model(model):
            raise ValueError(f"Model '{model}' is not supported by provider '{provider.value}'")
        
        return client
    
    def get_all_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models grouped by provider.
        
        Returns:
            Dictionary mapping provider names to their available models
        """
        models_by_provider = {}
        
        for provider in LLMProvider:
            try:
                # Create client if not exists
                if provider not in self.clients:
                    self.create_client(provider)
                
                client = self.clients[provider]
                models_by_provider[provider.value] = client.get_available_models()
                
            except Exception as e:
                # Skip providers that can't be initialized
                models_by_provider[provider.value] = []
        
        return models_by_provider
    
    def estimate_cost_for_model(self, model: str, prompt: str, max_tokens: Optional[int] = None) -> float:
        """
        Estimate cost for a request using a specific model.
        
        This can be used by Layer 1 for cost-based model selection.
        
        Args:
            model: Model to estimate cost for
            prompt: Prompt text
            max_tokens: Maximum tokens for completion
            
        Returns:
            Estimated cost in USD
        """
        try:
            client = self.get_client_for_model(model)
            from .llm_client import LLMRequest
            
            request = LLMRequest(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens
            )
            
            return client.estimate_cost(request)
            
        except Exception:
            return 0.0  # Return 0 if estimation fails
    
    def select_best_model(
        self, 
        task_type: str,
        complexity_level: str = "medium",
        budget_constraint: Optional[float] = None,
        latency_requirement: str = "normal"
    ) -> str:
        """
        Select the best model based on task requirements.
        
        This implements basic Layer 1 model selection logic.
        
        Args:
            task_type: Type of task (e.g., "code_generation", "qa", "analysis")
            complexity_level: Complexity level ("simple", "medium", "complex")
            budget_constraint: Maximum cost per request in USD
            latency_requirement: Latency requirement ("fast", "normal", "slow")
            
        Returns:
            Selected model name
        """
        # Define model capabilities and characteristics
        model_profiles = {
            "gpt-4": {
                "cost": "high", "performance": "high", "latency": "slow",
                "specialties": ["complex_reasoning", "code_generation", "analysis"]
            },
            "gpt-4-turbo": {
                "cost": "medium", "performance": "high", "latency": "normal",
                "specialties": ["complex_reasoning", "code_generation", "analysis"]
            },
            "gpt-3.5-turbo": {
                "cost": "low", "performance": "medium", "latency": "fast",
                "specialties": ["qa", "simple_reasoning", "general_tasks"]
            },
            "claude-3-opus-20240229": {
                "cost": "high", "performance": "high", "latency": "slow",
                "specialties": ["analysis", "writing", "complex_reasoning"]
            },
            "claude-3-sonnet-20240229": {
                "cost": "medium", "performance": "high", "latency": "normal",
                "specialties": ["analysis", "code_generation", "reasoning"]
            },
            "claude-3-haiku-20240307": {
                "cost": "low", "performance": "medium", "latency": "fast",
                "specialties": ["qa", "simple_tasks", "fast_response"]
            }
        }
        
        # Filter models based on constraints
        candidates = []
        
        for model, profile in model_profiles.items():
            # Check latency requirement
            if latency_requirement == "fast" and profile["latency"] not in ["fast", "normal"]:
                continue
            elif latency_requirement == "normal" and profile["latency"] == "slow":
                continue
            
            # Check if model specializes in this task type
            if task_type in profile["specialties"] or "general_tasks" in profile["specialties"]:
                candidates.append(model)
        
        # If no specialists found, use general models
        if not candidates:
            candidates = ["gpt-3.5-turbo", "claude-3-haiku-20240307"]
        
        # Select based on complexity level
        if complexity_level == "simple":
            # Prefer cheaper, faster models
            preferred_order = ["gpt-3.5-turbo", "claude-3-haiku-20240307", "gpt-4-turbo"]
        elif complexity_level == "complex":
            # Prefer more capable models
            preferred_order = ["gpt-4", "claude-3-opus-20240229", "gpt-4-turbo", "claude-3-sonnet-20240229"]
        else:  # medium
            # Balance of capability and cost
            preferred_order = ["gpt-4-turbo", "claude-3-sonnet-20240229", "gpt-4", "gpt-3.5-turbo"]
        
        # Select first available model from preferred order
        for model in preferred_order:
            if model in candidates:
                # Check budget constraint if specified
                if budget_constraint is not None:
                    estimated_cost = self.estimate_cost_for_model(model, "test prompt", 1000)
                    if estimated_cost > budget_constraint:
                        continue
                
                return model
        
        # Fallback to first candidate
        return candidates[0] if candidates else "gpt-3.5-turbo"
    
    def _get_api_key_from_env(self, provider: LLMProvider) -> Optional[str]:
        """Get API key from environment variables"""
        env_var_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OLLAMA: None,  # Ollama doesn't require API key
            LLMProvider.LITELLM: "LITELLM_API_KEY"
        }
        
        return os.getenv(env_var_map.get(provider))
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all clients"""
        stats = {}
        for provider, client in self.clients.items():
            stats[provider.value] = client.get_performance_stats()
        return stats
    
    def reset_all_stats(self):
        """Reset performance statistics for all clients"""
        for client in self.clients.values():
            client.reset_stats()