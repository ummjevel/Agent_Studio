"""
Abstract LLM client interface and base classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMRequest(BaseModel):
    """Request structure for LLM calls"""
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    system_message: Optional[str] = None
    
    # Prompt processing configuration (from Layer 2)
    use_cot: bool = False
    use_self_refine: bool = False
    use_self_consistency: bool = False
    meta_prompt: bool = False
    
    # Additional metadata
    task_type: Optional[str] = None
    complexity_level: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LLMResponse(BaseModel):
    """Response structure from LLM calls"""
    content: str
    model: str
    provider: str
    
    # Usage statistics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Performance metrics
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    
    # Quality indicators
    confidence_score: Optional[float] = None
    finish_reason: Optional[str] = None
    
    # Metadata
    timestamp: datetime = datetime.now()
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    This provides a unified interface for different LLM providers
    and integrates with Layer 1 (Model Selection) and Layer 2 (Prompt Preprocessing).
    """
    
    def __init__(self, provider: LLMProvider, api_key: Optional[str] = None, **kwargs):
        self.provider = provider
        self.api_key = api_key
        self.config = kwargs
        
        # Performance tracking
        self.request_count = 0
        self.total_cost = 0.0
        self.error_count = 0
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Complete a chat/text generation request.
        
        Args:
            request: LLM request with prompt and configuration
            
        Returns:
            LLM response with generated content and metadata
        """
        pass
    
    @abstractmethod
    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """
        Synchronous version of complete().
        
        Args:
            request: LLM request with prompt and configuration
            
        Returns:
            LLM response with generated content and metadata
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """
        Estimate the cost of a request in USD.
        
        Args:
            request: LLM request to estimate cost for
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is available and accessible.
        
        Args:
            model: Model name to validate
            
        Returns:
            True if model is valid and accessible
        """
        pass
    
    def apply_prompt_processing(self, request: LLMRequest) -> LLMRequest:
        """
        Apply Layer 2 prompt processing techniques.
        
        This integrates with the prompt preprocessing layer to enhance
        the prompt before sending to the LLM.
        """
        processed_request = request.model_copy()
        
        # Apply Chain-of-Thought if requested
        if request.use_cot:
            processed_request.prompt = self._apply_cot(request.prompt)
        
        # Apply Self-Refinement if requested
        if request.use_self_refine:
            processed_request.prompt = self._apply_self_refine(request.prompt)
        
        # Apply Meta-prompting if requested
        if request.meta_prompt:
            processed_request.prompt = self._apply_meta_prompting(request.prompt)
        
        return processed_request
    
    def _apply_cot(self, prompt: str) -> str:
        """Apply Chain-of-Thought prompting"""
        cot_prompt = f"""
Let's approach this step-by-step:

{prompt}

Please think through this systematically:
1. First, analyze what is being asked
2. Break down the problem into components
3. Work through each component
4. Combine the results into a comprehensive response

Your step-by-step reasoning:
"""
        return cot_prompt
    
    def _apply_self_refine(self, prompt: str) -> str:
        """Apply self-refinement prompting"""
        refine_prompt = f"""
{prompt}

After providing your initial response, please:
1. Review your answer for accuracy and completeness
2. Identify any potential improvements
3. Provide a refined, improved version if necessary

Initial response:
[Provide your response here]

Review and refinement:
[Review your response and provide improvements if needed]
"""
        return refine_prompt
    
    def _apply_meta_prompting(self, prompt: str) -> str:
        """Apply meta-prompting technique"""
        meta_prompt = f"""
You are an expert at understanding and responding to complex requests. 

Before responding to the following request, first:
1. Analyze what type of response would be most helpful
2. Consider what format and structure would be clearest
3. Think about what additional context might be useful

Request: {prompt}

Meta-analysis:
[Analyze the request and optimal response approach]

Response:
[Provide your carefully structured response]
"""
        return meta_prompt
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this client"""
        return {
            "provider": self.provider.value,
            "request_count": self.request_count,
            "total_cost_usd": self.total_cost,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_cost_per_request": self.total_cost / max(self.request_count, 1)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.request_count = 0
        self.total_cost = 0.0
        self.error_count = 0