"""
Concrete implementations of LLM providers
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import json

from .llm_client import LLMClient, LLMProvider, LLMRequest, LLMResponse


class OpenAIProvider(LLMClient):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(LLMProvider.OPENAI, api_key, **kwargs)
        
        # Model pricing (per 1K tokens) - approximate values
        self.model_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
        }
        
        self.available_models = list(self.model_pricing.keys())
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Async completion using OpenAI API"""
        try:
            # Import OpenAI here to avoid dependency issues
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = await client.chat.completions.create(
                model=processed_request.model,
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                frequency_penalty=processed_request.frequency_penalty,
                presence_penalty=processed_request.presence_penalty,
                stop=processed_request.stop
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost = self._calculate_cost(
                processed_request.model, 
                usage.prompt_tokens if usage else 0, 
                usage.completion_tokens if usage else 0
            )
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=response.id,
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion using OpenAI API"""
        try:
            # Import OpenAI here to avoid dependency issues
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=processed_request.model,
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                frequency_penalty=processed_request.frequency_penalty,
                presence_penalty=processed_request.presence_penalty,
                stop=processed_request.stop
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost = self._calculate_cost(
                processed_request.model, 
                usage.prompt_tokens if usage else 0, 
                usage.completion_tokens if usage else 0
            )
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=response.id,
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return self.available_models
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for OpenAI request"""
        if request.model not in self.model_pricing:
            return 0.0
        
        # Rough token estimation (4 chars = 1 token)
        estimated_prompt_tokens = len(request.prompt) // 4
        estimated_completion_tokens = request.max_tokens or 1000
        
        pricing = self.model_pricing[request.model]
        cost = (estimated_prompt_tokens * pricing["input"] / 1000) + \
               (estimated_completion_tokens * pricing["output"] / 1000)
        
        return cost
    
    def validate_model(self, model: str) -> bool:
        """Validate OpenAI model"""
        return model in self.available_models
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate actual cost based on usage"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        cost = (prompt_tokens * pricing["input"] / 1000) + \
               (completion_tokens * pricing["output"] / 1000)
        
        return cost


class AnthropicProvider(LLMClient):
    """Anthropic (Claude) API provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(LLMProvider.ANTHROPIC, api_key, **kwargs)
        
        # Model pricing (per 1K tokens) - approximate values
        self.model_pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024}
        }
        
        self.available_models = list(self.model_pricing.keys())
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Async completion using Anthropic API"""
        try:
            # Import Anthropic here to avoid dependency issues
            from anthropic import AsyncAnthropic
            
            client = AsyncAnthropic(api_key=self.api_key)
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = [{"role": "user", "content": processed_request.prompt}]
            
            # Make API call
            start_time = time.time()
            
            response = await client.messages.create(
                model=processed_request.model,
                max_tokens=processed_request.max_tokens or 4096,
                messages=messages,
                temperature=processed_request.temperature,
                top_p=processed_request.top_p,
                system=processed_request.system_message
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.content[0].text if response.content else ""
            
            # Calculate cost (Anthropic doesn't provide token counts in the same way)
            estimated_prompt_tokens = len(processed_request.prompt) // 4
            estimated_completion_tokens = len(content) // 4
            cost = self._calculate_cost(processed_request.model, estimated_prompt_tokens, estimated_completion_tokens)
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.stop_reason,
                request_id=response.id,
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion using Anthropic API"""
        try:
            # Import Anthropic here to avoid dependency issues
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = [{"role": "user", "content": processed_request.prompt}]
            
            # Make API call
            start_time = time.time()
            
            response = client.messages.create(
                model=processed_request.model,
                max_tokens=processed_request.max_tokens or 4096,
                messages=messages,
                temperature=processed_request.temperature,
                top_p=processed_request.top_p,
                system=processed_request.system_message
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.content[0].text if response.content else ""
            
            # Calculate cost
            estimated_prompt_tokens = len(processed_request.prompt) // 4
            estimated_completion_tokens = len(content) // 4
            cost = self._calculate_cost(processed_request.model, estimated_prompt_tokens, estimated_completion_tokens)
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.stop_reason,
                request_id=response.id,
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models"""
        return self.available_models
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Anthropic request"""
        if request.model not in self.model_pricing:
            return 0.0
        
        # Rough token estimation
        estimated_prompt_tokens = len(request.prompt) // 4
        estimated_completion_tokens = request.max_tokens or 1000
        
        pricing = self.model_pricing[request.model]
        cost = (estimated_prompt_tokens * pricing["input"] / 1000) + \
               (estimated_completion_tokens * pricing["output"] / 1000)
        
        return cost
    
    def validate_model(self, model: str) -> bool:
        """Validate Anthropic model"""
        return model in self.available_models
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate actual cost based on usage"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        cost = (prompt_tokens * pricing["input"] / 1000) + \
               (completion_tokens * pricing["output"] / 1000)
        
        return cost


class LiteLLMProvider(LLMClient):
    """LiteLLM provider for unified access to multiple LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(LLMProvider.LITELLM, api_key, **kwargs)
        
        # LiteLLM supports many models - this is a subset
        self.available_models = [
            # OpenAI
            "gpt-4", "gpt-3.5-turbo",
            # Anthropic
            "claude-2", "claude-instant-1",
            # Others via LiteLLM
            "gemini-pro", "palm-2", "cohere/command",
            # Open source via Hugging Face
            "huggingface/microsoft/DialoGPT-medium",
            "huggingface/facebook/blenderbot-400M-distill"
        ]
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Async completion using LiteLLM"""
        try:
            # Import LiteLLM here to avoid dependency issues
            import litellm
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = await litellm.acompletion(
                model=processed_request.model,
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                api_key=self.api_key
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Estimate cost (LiteLLM doesn't always provide costs)
            cost = 0.0  # Would need to implement cost calculation per provider
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=getattr(response, 'id', None),
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"LiteLLM API error: {str(e)}")
    
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion using LiteLLM"""
        try:
            # Import LiteLLM here to avoid dependency issues
            import litellm
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = litellm.completion(
                model=processed_request.model,
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                api_key=self.api_key
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Estimate cost
            cost = 0.0  # Would need to implement cost calculation per provider
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=getattr(response, 'id', None),
                metadata=processed_request.metadata
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"LiteLLM API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available models through LiteLLM"""
        return self.available_models
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for LiteLLM request"""
        # Would need to implement cost calculation per underlying provider
        return 0.0
    
    def validate_model(self, model: str) -> bool:
        """Validate model availability in LiteLLM"""
        return model in self.available_models


class AzureOpenAIProvider(LLMClient):
    """Azure OpenAI API provider implementation"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2024-02-15-preview",
                 **kwargs):
        super().__init__(LLMProvider.AZURE_OPENAI, api_key, **kwargs)
        
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        
        # Model pricing (same as OpenAI but may vary by region)
        self.model_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-35-turbo": {"input": 0.0015, "output": 0.002},  # Note: Azure uses gpt-35-turbo
            "gpt-35-turbo-16k": {"input": 0.003, "output": 0.004}
        }
        
        self.available_models = list(self.model_pricing.keys())
        
        # Azure deployment names mapping (can be customized)
        self.deployment_names = kwargs.get('deployment_names', {})
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Async completion using Azure OpenAI API"""
        try:
            # Import Azure OpenAI here to avoid dependency issues
            from openai import AsyncAzureOpenAI
            
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Get deployment name (use model name if not mapped)
            deployment_name = self.deployment_names.get(
                processed_request.model, 
                processed_request.model
            )
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = await client.chat.completions.create(
                model=deployment_name,  # Use deployment name for Azure
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                frequency_penalty=processed_request.frequency_penalty,
                presence_penalty=processed_request.presence_penalty,
                stop=processed_request.stop
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost = self._calculate_cost(
                processed_request.model, 
                usage.prompt_tokens if usage else 0, 
                usage.completion_tokens if usage else 0
            )
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=response.id,
                metadata={
                    **processed_request.metadata,
                    "azure_endpoint": self.azure_endpoint,
                    "deployment_name": deployment_name
                }
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Azure OpenAI API error: {str(e)}")
    
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion using Azure OpenAI API"""
        try:
            # Import Azure OpenAI here to avoid dependency issues
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Get deployment name
            deployment_name = self.deployment_names.get(
                processed_request.model, 
                processed_request.model
            )
            
            # Prepare messages
            messages = []
            if processed_request.system_message:
                messages.append({"role": "system", "content": processed_request.system_message})
            messages.append({"role": "user", "content": processed_request.prompt})
            
            # Make API call
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=processed_request.temperature,
                max_tokens=processed_request.max_tokens,
                top_p=processed_request.top_p,
                frequency_penalty=processed_request.frequency_penalty,
                presence_penalty=processed_request.presence_penalty,
                stop=processed_request.stop
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost = self._calculate_cost(
                processed_request.model, 
                usage.prompt_tokens if usage else 0, 
                usage.completion_tokens if usage else 0
            )
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
                request_id=response.id,
                metadata={
                    **processed_request.metadata,
                    "azure_endpoint": self.azure_endpoint,
                    "deployment_name": deployment_name
                }
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Azure OpenAI API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available Azure OpenAI models"""
        return self.available_models
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Azure OpenAI request"""
        if request.model not in self.model_pricing:
            return 0.0
        
        # Rough token estimation (4 chars = 1 token)
        estimated_prompt_tokens = len(request.prompt) // 4
        estimated_completion_tokens = request.max_tokens or 1000
        
        pricing = self.model_pricing[request.model]
        cost = (estimated_prompt_tokens * pricing["input"] / 1000) + \
               (estimated_completion_tokens * pricing["output"] / 1000)
        
        return cost
    
    def validate_model(self, model: str) -> bool:
        """Validate Azure OpenAI model"""
        return model in self.available_models
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate actual cost based on usage"""
        if model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[model]
        cost = (prompt_tokens * pricing["input"] / 1000) + \
               (completion_tokens * pricing["output"] / 1000)
        
        return cost
    
    def set_deployment_mapping(self, model_to_deployment: Dict[str, str]):
        """
        Set custom mapping from model names to Azure deployment names.
        
        Args:
            model_to_deployment: Dictionary mapping model names to deployment names
            
        Example:
            provider.set_deployment_mapping({
                "gpt-4": "my-gpt4-deployment",
                "gpt-35-turbo": "my-gpt35-deployment"
            })
        """
        self.deployment_names.update(model_to_deployment)


class OllamaProvider(LLMClient):
    """Ollama local LLM provider implementation"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "http://localhost:11434",
                 **kwargs):
        super().__init__(LLMProvider.OLLAMA, api_key, **kwargs)
        
        self.base_url = base_url
        
        # Ollama models are typically free to run locally
        self.model_pricing = {}  # No cost for local models
        
        # Common Ollama models
        self.available_models = [
            "llama2", "llama2:13b", "llama2:70b",
            "codellama", "codellama:13b", "codellama:34b",
            "mistral", "mistral:7b", "mixtral:8x7b",
            "neural-chat", "starling-lm", "dolphin-phi",
            "phi", "orca-mini", "vicuna",
            "gemma:2b", "gemma:7b"
        ]
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Async completion using Ollama API"""
        try:
            import aiohttp
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare request data
            data = {
                "model": processed_request.model,
                "prompt": processed_request.prompt,
                "stream": False,
                "options": {
                    "temperature": processed_request.temperature,
                    "top_p": processed_request.top_p,
                    "num_predict": processed_request.max_tokens or 1000
                }
            }
            
            if processed_request.system_message:
                data["system"] = processed_request.system_message
            
            # Make API call
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API returned status {response.status}")
                    
                    result = await response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = result.get("response", "")
            
            # Estimate token counts (Ollama doesn't always provide exact counts)
            estimated_prompt_tokens = len(processed_request.prompt) // 4
            estimated_completion_tokens = len(content) // 4
            
            # Update stats
            self.request_count += 1
            # No cost for local models
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Free for local models
                finish_reason=result.get("done_reason", "stop"),
                metadata={
                    **processed_request.metadata,
                    "ollama_base_url": self.base_url,
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count")
                }
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Ollama API error: {str(e)}")
    
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion using Ollama API"""
        try:
            import requests
            
            # Apply prompt processing
            processed_request = self.apply_prompt_processing(request)
            
            # Prepare request data
            data = {
                "model": processed_request.model,
                "prompt": processed_request.prompt,
                "stream": False,
                "options": {
                    "temperature": processed_request.temperature,
                    "top_p": processed_request.top_p,
                    "num_predict": processed_request.max_tokens or 1000
                }
            }
            
            if processed_request.system_message:
                data["system"] = processed_request.system_message
            
            # Make API call
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}")
            
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            content = result.get("response", "")
            
            # Estimate token counts
            estimated_prompt_tokens = len(processed_request.prompt) // 4
            estimated_completion_tokens = len(content) // 4
            
            # Update stats
            self.request_count += 1
            # No cost for local models
            
            return LLMResponse(
                content=content,
                model=processed_request.model,
                provider=self.provider.value,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Free for local models
                finish_reason=result.get("done_reason", "stop"),
                metadata={
                    **processed_request.metadata,
                    "ollama_base_url": self.base_url,
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count")
                }
            )
            
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Ollama API error: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            import requests
            
            # Try to get models from Ollama API
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
        except Exception:
            pass
        
        # Fallback to default list
        return self.available_models
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Ollama request (always 0 for local models)"""
        return 0.0
    
    def validate_model(self, model: str) -> bool:
        """Validate Ollama model availability"""
        available_models = self.get_available_models()
        return model in available_models
    
    def list_local_models(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about locally available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            import requests
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
        
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama repository.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            data = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=600  # 10 minute timeout for model downloads
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False