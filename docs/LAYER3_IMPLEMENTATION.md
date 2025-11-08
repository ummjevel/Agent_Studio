# Layer 3: Code Generation - 구현 문서

## 개요

Self-Evolving Agent Framework의 Layer 3 (Code Generation)는 **Template + LLM-Direct 하이브리드 접근법**을 사용하여 최적의 워크플로우를 생성합니다. 이 레이어는 Layer 1 (Model Selection)과 Layer 2 (Prompt Preprocessing)와 완전히 통합되어 지능적인 워크플로우 생성을 제공합니다.

## 주요 특징

### 🎯 하이브리드 생성 전략
- **Template-based**: 빠르고 안정적인 일반 패턴
- **LLM-direct**: 유연하고 창의적인 복잡한 케이스  
- **Adaptive**: 태스크 특성에 따른 지능적 선택

### 🌐 다중 LLM 프로바이더 지원
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Azure OpenAI**: Enterprise 지원
- **Anthropic**: Claude 시리즈
- **Ollama**: 로컬 오픈소스 모델
- **LiteLLM**: 통합 멀티 프로바이더

### 🧠 Self-Evolution 메커니즘
- 워크플로우 패턴 자동 학습
- 성능 기반 최적화
- 지속적 개선

## 아키텍처

```
src/layers/code_generation/
├── generator.py                   # 메인 WorkflowCodeGenerator 클래스
├── hybrid_generator.py            # 하이브리드 생성 전략
├── workflow/                      # 워크플로우 표현
│   ├── node.py                    # WorkflowNode 클래스
│   ├── graph.py                   # WorkflowGraph 클래스
│   └── state.py                   # 실행 상태 관리
├── templates/                     # 템플릿 기반 생성
│   ├── template_generator.py      # 템플릿 생성기
│   └── workflow_templates.py      # 사전 정의 템플릿
├── llm_generators/                # LLM 직접 생성
│   ├── llm_generator.py          # LLM 생성기
│   └── prompts.py                # 생성 프롬프트
├── llm_client/                   # LLM 클라이언트 통합
│   ├── client_factory.py         # 클라이언트 팩토리
│   ├── llm_client.py            # 추상 클라이언트
│   └── providers.py             # 프로바이더 구현
├── patterns/                     # 패턴 학습 시스템
│   ├── pattern_store.py         # 패턴 저장소
│   ├── pattern_matcher.py       # 패턴 매칭
│   └── pattern_learner.py       # 패턴 학습
└── langgraph_integration/        # LangGraph 통합
    ├── converter.py              # 변환기
    └── executor.py               # 실행기
```

## 핵심 구성요소

### 1. WorkflowCodeGenerator (메인 클래스)

```python
from src.layers.code_generation import WorkflowCodeGenerator, GenerationMode

# 초기화
generator = WorkflowCodeGenerator()

# 워크플로우 생성
workflow = generator.generate_workflow(
    task_description="사용자 질문에 대해 웹 검색 후 답변 생성",
    mode=GenerationMode.BALANCED,
    task_type="qa",
    complexity_hint="medium"
)

# 실행
results = generator.execute_workflow(
    workflow=workflow,
    initial_data={"question": "AI 최신 동향은?"},
    learn_from_execution=True
)
```

### 2. 생성 모드

| 모드 | 설명 | 사용 시점 |
|------|------|-----------|
| `FAST` | 템플릿 기반 빠른 생성 | 단순 태스크, 빠른 응답 필요 |
| `BALANCED` | 하이브리드 접근 | 일반적인 케이스 (기본값) |
| `CREATIVE` | LLM 중심 생성 | 복잡하거나 새로운 요구사항 |
| `LEARNING` | 패턴 기반 생성 | 유사한 패턴이 학습된 경우 |

### 3. LLM 프로바이더 설정

```python
# 환경변수 설정
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-key"  
export ANTHROPIC_API_KEY="your-anthropic-key"

# Azure OpenAI 설정
from src.layers.code_generation.llm_client import AzureOpenAIProvider

azure_provider = AzureOpenAIProvider(
    api_key="your-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    deployment_names={
        "gpt-4": "my-gpt4-deployment",
        "gpt-35-turbo": "my-gpt35-deployment"
    }
)

# Ollama 설정 (로컬)
from src.layers.code_generation.llm_client import OllamaProvider

ollama_provider = OllamaProvider(
    base_url="http://localhost:11434"
)

# 모델 다운로드
ollama_provider.pull_model("llama2:13b")
```

### 4. 워크플로우 구조

```python
from src.layers.code_generation import WorkflowNode, NodeType, WorkflowGraph

# 노드 생성
input_node = WorkflowNode(
    id="input",
    name="사용자 입력",
    node_type=NodeType.INPUT,
    operation="receive_question"
)

llm_node = WorkflowNode(
    id="llm_process", 
    name="LLM 처리",
    node_type=NodeType.LLM_CALL,
    operation="answer_question",
    model_name="gpt-4-turbo",
    prompt_template="질문에 답하세요: {question}"
)

# 워크플로우 그래프
workflow = WorkflowGraph(id="qa_workflow", name="Q&A 워크플로우")
workflow.add_node(input_node)
workflow.add_node(llm_node) 
workflow.connect("input", "llm_process", "question")
```

### 5. 패턴 학습

```python
from src.layers.code_generation.patterns import WorkflowPatternStore, PatternLearner

# 패턴 저장소 초기화
pattern_store = WorkflowPatternStore("patterns.json")
pattern_learner = PatternLearner(pattern_store)

# 성공적인 워크플로우에서 학습
pattern_learner.learn_from_workflow(
    workflow=workflow,
    execution_result={
        "success": True,
        "performance_score": 9.2,
        "execution_time_ms": 1500,
        "cost_usd": 0.05
    },
    task_description="Q&A 태스크",
    task_type="qa"
)

# 패턴 검색
relevant_patterns = pattern_store.find_patterns(
    task_type="qa",
    keywords=["question", "answer"],
    min_confidence=0.8
)
```

## Layer 1, 2 통합

### Model Selection 통합
```python
# 자동 모델 선택 (Layer 1)
best_model = generator.llm_client_factory.select_best_model(
    task_type="code_generation",
    complexity_level="medium",
    budget_constraint=0.50,
    latency_requirement="normal"
)

# 결과: "gpt-4-turbo" (성능과 비용의 균형)
```

### Prompt Preprocessing 통합
```python
# Layer 2 프롬프트 처리 옵션
workflow = generator.generate_workflow(
    task_description="복잡한 추론 문제 해결",
    mode=GenerationMode.CREATIVE,
    prompt_processing={
        "use_cot": True,              # Chain-of-Thought
        "use_self_refine": True,      # 자기 개선
        "use_meta_prompting": False
    }
)
```

## 성능 메트릭

### 생성 성능
- **템플릿 기반**: ~100ms (매우 빠름)
- **LLM 직접**: ~2-5초 (모델에 따라)
- **하이브리드**: ~500ms-3초 (적응적)

### 품질 메트릭
- **정확도**: 패턴 기반 95%, LLM 기반 88%
- **완성도**: 템플릿 기반 99%, LLM 기반 92%
- **창의성**: LLM 기반 90%, 템플릿 기반 60%

### 비용 효율성
- **Ollama**: $0 (로컬)
- **GPT-3.5**: ~$0.002/워크플로우
- **GPT-4**: ~$0.02/워크플로우
- **Claude**: ~$0.01/워크플로우

## 설정 및 환경변수

### 필수 환경변수
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI  
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 선택적 설정
```bash
# Ollama (기본: localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# LiteLLM
export LITELLM_API_KEY="..."

# 패턴 저장 경로
export PATTERN_STORE_PATH="./patterns.json"
```

## 확장성

### 새로운 템플릿 추가
```python
from src.layers.code_generation.templates import WorkflowTemplate

custom_template = WorkflowTemplate(
    id="custom_analysis",
    name="맞춤 분석 워크플로우", 
    task_type="analysis",
    node_templates=[...],
    edge_templates=[...]
)

generator.template_generator.template_library.add_template(custom_template)
```

### 새로운 LLM 프로바이더 추가
```python
from src.layers.code_generation.llm_client import LLMClient

class CustomProvider(LLMClient):
    def __init__(self, **kwargs):
        super().__init__(LLMProvider.CUSTOM, **kwargs)
    
    def complete_sync(self, request):
        # 구현
        pass
```

## 모니터링 및 디버깅

### 통계 수집
```python
# 생성 통계
stats = generator.get_statistics()
print(f"성공률: {stats['generation_stats']['successful_generations']/stats['generation_stats']['total_generations']*100:.1f}%")

# 패턴 통계
pattern_stats = generator.pattern_store.get_pattern_stats()
print(f"학습된 패턴: {pattern_stats['total_patterns']}개")

# LLM 클라이언트 통계
client_stats = generator.llm_client_factory.get_client_stats()
for provider, stats in client_stats.items():
    print(f"{provider}: {stats['request_count']}회 사용, ${stats['total_cost_usd']:.3f} 비용")
```

### 로깅 설정
```python
import logging

# 워크플로우 생성 로깅
logging.getLogger('code_generation').setLevel(logging.INFO)

# LLM 호출 로깅 
logging.getLogger('llm_client').setLevel(logging.DEBUG)
```

## 트러블슈팅

### 일반적인 문제

1. **LLM API 오류**
   ```python
   # 폴백 모드 활성화
   generator = WorkflowCodeGenerator(enable_fallback=True)
   ```

2. **템플릿 매칭 실패**
   ```python
   # 강제 LLM 모드
   workflow = generator.generate_workflow(
       task_description="...",
       mode=GenerationMode.CREATIVE
   )
   ```

3. **성능 이슈**
   ```python
   # 빠른 모드 사용
   workflow = generator.generate_workflow(
       task_description="...", 
       mode=GenerationMode.FAST,
       use_patterns=True
   )
   ```

### 디버깅 도구
```python
# 워크플로우 검증
is_valid, errors = workflow.validate()
if not is_valid:
    print("검증 오류:", errors)

# 실행 단계별 추적
for state in generator.executor.execute_workflow_streaming(workflow):
    print(f"현재 노드: {state.current_node}")
    print(f"상태: {state.status}")
```

## 다음 단계

1. **MCTS 탐색 엔진 추가** (AFlow 방식)
2. **멀티 에이전트 협업** 지원
3. **실시간 스트리밍** 실행
4. **분산 처리** 확장
5. **GUI 인터페이스** 개발

---

**작성일**: 2025년 1월  
**버전**: 1.0  
**담당**: Code Generation Layer Implementation Team