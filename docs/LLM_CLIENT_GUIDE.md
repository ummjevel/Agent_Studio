# LLM Client 사용 가이드

## 개요

Self-Evolving Agent Framework의 LLM Client 시스템은 다양한 LLM 프로바이더를 통합하여 일관된 인터페이스를 제공합니다. 이 가이드는 각 프로바이더의 설정과 사용법을 상세히 설명합니다.

## 지원 프로바이더

### 1. OpenAI

#### 기본 설정
```python
from src.layers.code_generation.llm_client import OpenAIProvider

# API Key 설정
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# 클라이언트 생성
provider = OpenAIProvider()

# 모델 호출
from src.layers.code_generation.llm_client.llm_client import LLMRequest

request = LLMRequest(
    prompt="Python에서 리스트를 정렬하는 방법을 알려주세요.",
    model="gpt-4-turbo",
    max_tokens=1000,
    temperature=0.7
)

response = provider.complete_sync(request)
print(response.content)
```

#### 지원 모델
- `gpt-4`: 가장 강력한 모델, 복잡한 추론과 코드 생성
- `gpt-4-turbo`: GPT-4의 빠른 버전, 비용 효율적
- `gpt-3.5-turbo`: 빠르고 저렴한 범용 모델
- `gpt-3.5-turbo-16k`: 긴 컨텍스트 지원

### 2. Azure OpenAI

#### 기본 설정
```python
from src.layers.code_generation.llm_client import AzureOpenAIProvider

# 환경변수 설정
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"

# 클라이언트 생성 (배포명 매핑 필요)
provider = AzureOpenAIProvider(
    deployment_names={
        "gpt-4": "my-gpt4-deployment",
        "gpt-35-turbo": "my-gpt35-deployment"
    }
)

# 모델 호출 (Azure 모델명 사용)
request = LLMRequest(
    prompt="데이터베이스 설계 패턴을 설명해주세요.",
    model="gpt-35-turbo",  # Azure 모델명
    max_tokens=800
)

response = provider.complete_sync(request)
```

#### Azure 특화 설정
```python
# 커스텀 API 버전 사용
provider = AzureOpenAIProvider(
    api_version="2024-02-15-preview",
    deployment_names={
        "gpt-4": "prod-gpt4-deployment",
        "gpt-35-turbo": "dev-gpt35-deployment"
    }
)

# 다중 리전 설정
providers = {
    "east": AzureOpenAIProvider(
        azure_endpoint="https://east-resource.openai.azure.com/",
        deployment_names={"gpt-4": "east-gpt4"}
    ),
    "west": AzureOpenAIProvider(
        azure_endpoint="https://west-resource.openai.azure.com/",
        deployment_names={"gpt-4": "west-gpt4"}
    )
}
```

### 3. Anthropic (Claude)

#### 기본 설정
```python
from src.layers.code_generation.llm_client import AnthropicProvider

# API Key 설정
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# 클라이언트 생성
provider = AnthropicProvider()

# Claude 모델 호출
request = LLMRequest(
    prompt="함수형 프로그래밍의 핵심 개념을 설명해주세요.",
    model="claude-3-sonnet-20240229",
    max_tokens=1500,
    temperature=0.3
)

response = provider.complete_sync(request)
```

#### 지원 모델
- `claude-3-opus-20240229`: 가장 강력한 Claude 모델
- `claude-3-sonnet-20240229`: 균형 잡힌 성능과 속도
- `claude-3-haiku-20240307`: 빠르고 경제적인 모델
- `claude-2.1`: 이전 버전, 안정성 우선시

### 4. Ollama (로컬 모델)

#### 설치 및 설정
```bash
# Ollama 설치 (macOS)
brew install ollama

# Ollama 서버 시작
ollama serve

# 모델 다운로드
ollama pull llama2:13b
ollama pull codellama:34b
ollama pull mistral:7b
```

#### Python 사용법
```python
from src.layers.code_generation.llm_client import OllamaProvider

# 클라이언트 생성 (기본 URL: http://localhost:11434)
provider = OllamaProvider()

# 사용 가능한 모델 확인
models = provider.get_available_models()
print("설치된 모델:", models)

# 모델 호출
request = LLMRequest(
    prompt="파이썬으로 웹 스크래핑하는 코드를 작성해주세요.",
    model="codellama:34b",
    max_tokens=2000,
    temperature=0.1
)

response = provider.complete_sync(request)
```

#### 모델 관리
```python
# 새 모델 다운로드
provider.pull_model("mixtral:8x7b")

# 모델 삭제
provider.remove_model("llama2:7b")

# 모델 정보 확인
model_info = provider.get_model_info("codellama:34b")
print(f"모델 크기: {model_info.get('size', 'N/A')}")
```

### 5. LiteLLM (통합 프로바이더)

#### 기본 설정
```python
from src.layers.code_generation.llm_client import LiteLLMProvider

# 다양한 프로바이더의 모델 사용
provider = LiteLLMProvider()

# Google Gemini 사용
request = LLMRequest(
    prompt="머신러닝 모델 평가 지표를 설명해주세요.",
    model="gemini-pro",
    max_tokens=1200
)

response = provider.complete_sync(request)
```

#### 지원 모델 예시
```python
# Hugging Face 모델
request = LLMRequest(
    prompt="자연어 처리 태스크를 분류해주세요.",
    model="huggingface/microsoft/DialoGPT-medium",
    max_tokens=800
)

# Cohere 모델
request = LLMRequest(
    prompt="텍스트 분류 모델을 설계해주세요.",
    model="cohere/command",
    max_tokens=1000
)
```

## LLMClientFactory 사용법

### 자동 모델 선택
```python
from src.layers.code_generation.llm_client import LLMClientFactory

# 팩토리 초기화
factory = LLMClientFactory()

# 태스크에 최적화된 모델 자동 선택
best_model = factory.select_best_model(
    task_type="code_generation",
    complexity_level="medium",
    budget_constraint=0.10,  # $0.10 제한
    latency_requirement="fast"
)

print(f"선택된 모델: {best_model}")  # 예: "gpt-3.5-turbo"

# 선택된 모델의 클라이언트 자동 생성
client = factory.get_client_for_model(best_model)
```

### 비용 추정
```python
# 요청 전 비용 추정
estimated_cost = factory.estimate_cost_for_model(
    model="gpt-4",
    prompt="매우 긴 프롬프트 텍스트...",
    max_tokens=2000
)

print(f"예상 비용: ${estimated_cost:.4f}")

# 예산 내에서 최적 모델 선택
if estimated_cost > 0.05:
    # 더 경제적인 모델로 fallback
    best_model = factory.select_best_model(
        task_type="qa",
        complexity_level="simple",
        budget_constraint=0.05
    )
```

### 성능 모니터링
```python
# 클라이언트 성능 통계 수집
stats = factory.get_client_stats()

for provider, provider_stats in stats.items():
    print(f"{provider}:")
    print(f"  요청 수: {provider_stats['request_count']}")
    print(f"  총 비용: ${provider_stats['total_cost_usd']:.3f}")
    print(f"  평균 응답 시간: {provider_stats['avg_response_time_ms']:.0f}ms")
    print(f"  성공률: {provider_stats['success_rate']*100:.1f}%")
```

## 고급 사용법

### 비동기 처리
```python
import asyncio

async def async_llm_call():
    provider = OpenAIProvider()
    
    # 비동기 요청
    request = LLMRequest(
        prompt="비동기 처리에 대해 설명해주세요.",
        model="gpt-3.5-turbo"
    )
    
    response = await provider.complete_async(request)
    return response.content

# 실행
result = asyncio.run(async_llm_call())
```

### 배치 처리
```python
async def batch_processing():
    provider = OpenAIProvider()
    
    # 여러 요청을 동시에 처리
    requests = [
        LLMRequest(prompt=f"주제 {i}에 대해 설명해주세요.", model="gpt-3.5-turbo")
        for i in range(1, 6)
    ]
    
    # 동시 실행
    tasks = [provider.complete_async(req) for req in requests]
    responses = await asyncio.gather(*tasks)
    
    return [resp.content for resp in responses]

results = asyncio.run(batch_processing())
```

### 에러 핸들링
```python
from src.layers.code_generation.llm_client.llm_client import LLMError

try:
    response = provider.complete_sync(request)
except LLMError as e:
    print(f"LLM 에러: {e.message}")
    print(f"에러 타입: {e.error_type}")
    print(f"재시도 가능: {e.retryable}")
    
    if e.retryable:
        # 재시도 로직
        import time
        time.sleep(1)
        response = provider.complete_sync(request)
```

### 커스텀 설정
```python
# 타임아웃 설정
provider = OpenAIProvider(
    request_timeout=30.0,  # 30초 타임아웃
    max_retries=3
)

# 로깅 설정
import logging
logging.getLogger('llm_client').setLevel(logging.DEBUG)

# 요청 시 추가 파라미터
request = LLMRequest(
    prompt="코드를 생성해주세요.",
    model="gpt-4",
    max_tokens=1000,
    temperature=0.2,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1
)
```

## 모범 사례

### 1. 모델 선택 전략
```python
# 태스크 유형별 권장 모델
TASK_MODEL_MAP = {
    "code_generation": ["gpt-4", "claude-3-sonnet-20240229", "codellama:34b"],
    "qa": ["gpt-3.5-turbo", "claude-3-haiku-20240307", "llama2:13b"],
    "analysis": ["gpt-4", "claude-3-opus-20240229"],
    "simple_tasks": ["gpt-3.5-turbo", "claude-3-haiku-20240307"]
}

def select_optimal_model(task_type: str, budget: float = None):
    candidates = TASK_MODEL_MAP.get(task_type, ["gpt-3.5-turbo"])
    
    if budget:
        # 예산 고려한 선택
        for model in candidates:
            cost = factory.estimate_cost_for_model(model, "test", 1000)
            if cost <= budget:
                return model
    
    return candidates[0]
```

### 2. 토큰 관리
```python
def optimize_prompt_for_model(prompt: str, model: str) -> str:
    """모델별 최적화된 프롬프트 생성"""
    
    if "claude" in model:
        # Claude는 구조화된 프롬프트 선호
        return f"<task>\n{prompt}\n</task>\n\n응답을 단계별로 제공해주세요."
    
    elif "gpt" in model:
        # GPT는 간결한 지시 선호
        return f"{prompt}\n\n단계별로 설명해주세요."
    
    else:
        # 기본 프롬프트
        return prompt
```

### 3. 에러 복구
```python
class ResilientLLMClient:
    def __init__(self):
        self.factory = LLMClientFactory()
        self.fallback_models = ["gpt-3.5-turbo", "claude-3-haiku-20240307"]
    
    def complete_with_fallback(self, request: LLMRequest):
        """에러 시 fallback 모델로 재시도"""
        
        # 원래 모델로 시도
        try:
            client = self.factory.get_client_for_model(request.model)
            return client.complete_sync(request)
        except LLMError as e:
            print(f"주 모델 실패: {e.message}")
        
        # Fallback 모델들로 순차 시도
        for fallback_model in self.fallback_models:
            try:
                request.model = fallback_model
                client = self.factory.get_client_for_model(fallback_model)
                return client.complete_sync(request)
            except LLMError:
                continue
        
        raise LLMError("모든 fallback 모델 실패", "COMPLETE_FAILURE")
```

## 트러블슈팅

### 자주 발생하는 문제

1. **API 키 오류**
   ```bash
   # 환경변수 확인
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   
   # 키 형식 검증
   python -c "import os; print(len(os.getenv('OPENAI_API_KEY', '')))"
   ```

2. **Ollama 연결 실패**
   ```bash
   # Ollama 서버 상태 확인
   curl http://localhost:11434/api/tags
   
   # 서버 재시작
   ollama serve
   ```

3. **Azure 배포명 오류**
   ```python
   # 배포명 확인
   provider = AzureOpenAIProvider()
   print(provider.deployment_names)
   
   # 올바른 배포명 설정
   provider.deployment_names["gpt-4"] = "actual-deployment-name"
   ```

4. **메모리 부족 (Ollama)**
   ```python
   # 작은 모델 사용
   request.model = "llama2:7b"  # 대신 llama2:70b
   
   # 컨텍스트 길이 제한
   request.max_tokens = 500
   ```

---

**작성일**: 2025년 1월  
**버전**: 1.0  
**담당**: LLM Client Implementation Team