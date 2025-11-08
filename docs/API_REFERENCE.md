# API 레퍼런스

## 개요

Self-Evolving Agent Framework Layer 3 (Code Generation)의 완전한 API 레퍼런스입니다. 모든 클래스, 메서드, 파라미터에 대한 상세한 정보를 제공합니다.

## 메인 API

### WorkflowCodeGenerator

워크플로우 생성의 메인 진입점입니다.

```python
class WorkflowCodeGenerator:
    def __init__(
        self,
        llm_client_factory: Optional[LLMClientFactory] = None,
        pattern_store_path: Optional[str] = None,
        enable_learning: bool = True,
        enable_fallback: bool = True
    )
```

**Parameters:**
- `llm_client_factory`: LLM 클라이언트 팩토리. None이면 기본 팩토리 생성
- `pattern_store_path`: 패턴 저장 파일 경로. None이면 "patterns.json" 사용
- `enable_learning`: 패턴 학습 활성화 여부 (기본값: True)
- `enable_fallback`: 오류 시 fallback 모드 사용 여부 (기본값: True)

#### 주요 메서드

##### generate_workflow()

```python
def generate_workflow(
    self,
    task_description: str,
    mode: Optional[GenerationMode] = None,
    task_type: Optional[str] = None,
    complexity_hint: str = "medium",
    required_tools: Optional[List[str]] = None,
    workflow_patterns: Optional[List[str]] = None,
    prompt_processing: Optional[Dict[str, Any]] = None,
    use_learned_patterns: bool = True,
    model_preferences: Optional[Dict[str, Any]] = None
) -> WorkflowGraph
```

워크플로우를 생성합니다.

**Parameters:**
- `task_description`: 수행할 작업에 대한 설명
- `mode`: 생성 모드 (FAST, BALANCED, CREATIVE, LEARNING)
- `task_type`: 작업 유형 ("qa", "analysis", "code_generation" 등)
- `complexity_hint`: 복잡도 힌트 ("simple", "medium", "complex")
- `required_tools`: 필요한 도구 목록
- `workflow_patterns`: 적용할 워크플로우 패턴
- `prompt_processing`: Layer 2 프롬프트 처리 옵션
- `use_learned_patterns`: 학습된 패턴 사용 여부
- `model_preferences`: 모델 선택 선호도

**Returns:**
- `WorkflowGraph`: 생성된 워크플로우 그래프

**Example:**
```python
generator = WorkflowCodeGenerator()

workflow = generator.generate_workflow(
    task_description="고객 리뷰 감정 분석",
    mode=GenerationMode.BALANCED,
    task_type="sentiment_analysis",
    complexity_hint="medium"
)
```

##### execute_workflow()

```python
def execute_workflow(
    self,
    workflow: WorkflowGraph,
    initial_data: Dict[str, Any],
    execution_mode: str = "sequential",
    learn_from_execution: bool = True,
    max_retries: int = 3,
    timeout_seconds: Optional[int] = None
) -> Dict[str, Any]
```

워크플로우를 실행합니다.

**Parameters:**
- `workflow`: 실행할 워크플로우
- `initial_data`: 초기 입력 데이터
- `execution_mode`: 실행 모드 ("sequential", "parallel", "adaptive")
- `learn_from_execution`: 실행 결과로부터 학습 여부
- `max_retries`: 최대 재시도 횟수
- `timeout_seconds`: 실행 제한 시간 (초)

**Returns:**
- `Dict[str, Any]`: 실행 결과

##### get_statistics()

```python
def get_statistics(self) -> Dict[str, Any]
```

생성기 통계 정보를 반환합니다.

**Returns:**
```python
{
    "generation_stats": {
        "total_generations": int,
        "successful_generations": int,
        "failed_generations": int,
        "avg_generation_time_ms": float
    },
    "execution_stats": {
        "total_executions": int,
        "successful_executions": int,
        "avg_execution_time_ms": float
    },
    "pattern_stats": {
        "total_patterns_learned": int,
        "patterns_used_count": int
    }
}
```

## 워크플로우 구조

### WorkflowGraph

워크플로우의 전체 구조를 나타내는 그래프 클래스입니다.

```python
class WorkflowGraph(BaseModel):
    id: str = Field(..., description="워크플로우 고유 ID")
    name: str = Field(..., description="워크플로우 이름")
    description: Optional[str] = Field(None, description="워크플로우 설명")
    nodes: List[WorkflowNode] = Field(default_factory=list, description="노드 목록")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="엣지 목록")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
```

#### 주요 메서드

##### add_node()

```python
def add_node(self, node: WorkflowNode) -> None
```

노드를 워크플로우에 추가합니다.

##### connect()

```python
def connect(
    self,
    from_node_id: str,
    to_node_id: str,
    data_key: str = "output",
    condition: Optional[str] = None
) -> None
```

두 노드를 연결합니다.

##### validate()

```python
def validate(self) -> Tuple[bool, List[str]]
```

워크플로우의 유효성을 검증합니다.

**Returns:**
- `Tuple[bool, List[str]]`: (유효성, 오류 목록)

##### get_execution_order()

```python
def get_execution_order(self) -> List[WorkflowNode]
```

실행 순서대로 정렬된 노드 목록을 반환합니다.

##### to_langgraph()

```python
def to_langgraph(self) -> Any
```

LangGraph 형식으로 변환합니다.

### WorkflowNode

개별 작업 단위를 나타내는 노드 클래스입니다.

```python
class WorkflowNode(BaseModel):
    id: str = Field(..., description="노드 고유 ID")
    name: str = Field(..., description="노드 이름")
    description: Optional[str] = Field(None, description="노드 설명")
    node_type: NodeType = Field(..., description="노드 타입")
    operation: str = Field(..., description="수행할 작업")
    
    # LLM 관련 필드
    model_name: Optional[str] = Field(None, description="사용할 LLM 모델")
    prompt_template: Optional[str] = Field(None, description="프롬프트 템플릿")
    max_tokens: Optional[int] = Field(None, description="최대 토큰 수")
    temperature: Optional[float] = Field(None, description="창의성 정도")
    
    # 검증 관련 필드
    validation_rules: Optional[List[Dict[str, Any]]] = Field(None, description="검증 규칙")
    
    # 결정 관련 필드
    decision_logic: Optional[Dict[str, Any]] = Field(None, description="결정 로직")
    
    # 메타데이터
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="실행 메타데이터")
```

### NodeType

노드 타입을 정의하는 열거형입니다.

```python
class NodeType(Enum):
    INPUT = "input"                    # 입력 노드
    OUTPUT = "output"                  # 출력 노드
    LLM_CALL = "llm_call"             # LLM 호출 노드
    TOOL_USE = "tool_use"             # 도구 사용 노드
    DECISION = "decision"             # 결정 노드
    LOOP = "loop"                     # 반복 노드
    AGGREGATION = "aggregation"       # 집계 노드
    VALIDATION = "validation"         # 검증 노드
    TRANSFORMATION = "transformation"  # 변환 노드
    CONDITION = "condition"           # 조건 노드
```

## 생성 모드

### GenerationMode

워크플로우 생성 전략을 정의하는 열거형입니다.

```python
class GenerationMode(Enum):
    FAST = "fast"           # 템플릿 기반 빠른 생성
    BALANCED = "balanced"   # 하이브리드 접근법
    CREATIVE = "creative"   # LLM 중심 창의적 생성
    LEARNING = "learning"   # 패턴 기반 학습 모드
```

## LLM 클라이언트

### LLMClientFactory

LLM 클라이언트를 생성하고 관리하는 팩토리 클래스입니다.

```python
class LLMClientFactory:
    def __init__(self)
```

#### 주요 메서드

##### create_client()

```python
def create_client(
    self,
    provider: LLMProvider,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient
```

LLM 클라이언트를 생성합니다.

##### get_client_for_model()

```python
def get_client_for_model(self, model: str) -> LLMClient
```

특정 모델에 대한 클라이언트를 반환합니다.

##### select_best_model()

```python
def select_best_model(
    self,
    task_type: str,
    complexity_level: str = "medium",
    budget_constraint: Optional[float] = None,
    latency_requirement: str = "normal"
) -> str
```

조건에 맞는 최적 모델을 선택합니다.

### LLMClient

LLM 클라이언트의 추상 기본 클래스입니다.

```python
class LLMClient(ABC):
    def __init__(self, provider: LLMProvider, **kwargs)
```

#### 주요 메서드

##### complete_sync()

```python
def complete_sync(self, request: LLMRequest) -> LLMResponse
```

동기적으로 LLM 요청을 처리합니다.

##### complete_async()

```python
async def complete_async(self, request: LLMRequest) -> LLMResponse
```

비동기적으로 LLM 요청을 처리합니다.

##### estimate_cost()

```python
def estimate_cost(self, request: LLMRequest) -> float
```

요청 비용을 추정합니다.

##### validate_model()

```python
def validate_model(self, model: str) -> bool
```

모델이 지원되는지 확인합니다.

### LLMRequest

LLM 요청을 나타내는 클래스입니다.

```python
class LLMRequest(BaseModel):
    prompt: str = Field(..., description="프롬프트 텍스트")
    model: str = Field(..., description="사용할 모델명")
    max_tokens: Optional[int] = Field(None, description="최대 토큰 수")
    temperature: Optional[float] = Field(None, description="창의성 정도 (0.0-2.0)")
    top_p: Optional[float] = Field(None, description="Top-p 샘플링")
    frequency_penalty: Optional[float] = Field(None, description="빈도 패널티")
    presence_penalty: Optional[float] = Field(None, description="존재 패널티")
    stop: Optional[List[str]] = Field(None, description="중지 토큰")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
```

### LLMResponse

LLM 응답을 나타내는 클래스입니다.

```python
class LLMResponse(BaseModel):
    content: str = Field(..., description="응답 내용")
    model: str = Field(..., description="사용된 모델")
    usage: Dict[str, int] = Field(..., description="토큰 사용량")
    cost_usd: Optional[float] = Field(None, description="비용 (USD)")
    response_time_ms: float = Field(..., description="응답 시간 (ms)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
```

### LLMProvider

LLM 프로바이더를 정의하는 열거형입니다.

```python
class LLMProvider(Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LITELLM = "litellm"
```

## 패턴 학습 시스템

### WorkflowPatternStore

패턴을 저장하고 관리하는 클래스입니다.

```python
class WorkflowPatternStore:
    def __init__(self, storage_path: str = "patterns.json")
```

#### 주요 메서드

##### add_pattern()

```python
def add_pattern(self, pattern: WorkflowPattern) -> None
```

새 패턴을 추가합니다.

##### find_patterns()

```python
def find_patterns(
    self,
    task_type: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    max_results: int = 10
) -> List[WorkflowPattern]
```

조건에 맞는 패턴들을 검색합니다.

##### get_pattern_stats()

```python
def get_pattern_stats(self) -> Dict[str, Any]
```

패턴 통계를 반환합니다.

### PatternLearner

실행 결과로부터 패턴을 학습하는 클래스입니다.

```python
class PatternLearner:
    def __init__(self, pattern_store: WorkflowPatternStore)
```

#### 주요 메서드

##### learn_from_workflow()

```python
def learn_from_workflow(
    self,
    workflow: WorkflowGraph,
    execution_result: Dict[str, Any],
    task_description: str,
    task_type: str,
    context_features: Optional[Dict[str, Any]] = None
) -> Optional[WorkflowPattern]
```

워크플로우 실행 결과로부터 패턴을 학습합니다.

### WorkflowPattern

학습된 워크플로우 패턴을 나타내는 클래스입니다.

```python
class WorkflowPattern(BaseModel):
    id: str = Field(..., description="패턴 고유 ID")
    name: str = Field(..., description="패턴 이름")
    description: Optional[str] = Field(None, description="패턴 설명")
    task_type: str = Field(..., description="태스크 타입")
    complexity_level: str = Field(..., description="복잡도")
    
    node_sequence: List[Dict[str, Any]] = Field(..., description="노드 시퀀스")
    connections: List[Dict[str, Any]] = Field(..., description="연결 정보")
    
    performance_metrics: Dict[str, float] = Field(..., description="성능 메트릭")
    applicable_conditions: List[Dict[str, Any]] = Field(..., description="적용 조건")
    learning_metadata: Dict[str, Any] = Field(..., description="학습 메타데이터")
    
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="수정 시간")
```

## 하이브리드 생성기

### HybridWorkflowGenerator

템플릿과 LLM을 조합한 하이브리드 생성기입니다.

```python
class HybridWorkflowGenerator:
    def __init__(self, llm_client_factory: LLMClientFactory)
```

#### 주요 메서드

##### generate_hybrid_workflow()

```python
def generate_hybrid_workflow(
    self,
    task_description: str,
    task_context: Optional[Dict[str, Any]] = None,
    strategy_hints: Optional[Dict[str, Any]] = None
) -> WorkflowGraph
```

하이브리드 전략으로 워크플로우를 생성합니다.

### WorkflowComplexityAnalyzer

작업의 복잡도를 분석하는 클래스입니다.

```python
class WorkflowComplexityAnalyzer:
    def analyze_complexity(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]
```

**Returns:**
```python
{
    "complexity_score": float,      # 0.0-1.0
    "complexity_level": str,        # "simple", "medium", "complex"
    "factors": {
        "task_scope": float,
        "data_complexity": float,
        "logic_complexity": float,
        "integration_complexity": float
    },
    "recommended_strategy": str,    # "template", "llm", "hybrid"
    "estimated_nodes": int,
    "estimated_execution_time": float
}
```

## 예외 및 에러 처리

### WorkflowGenerationError

워크플로우 생성 관련 예외입니다.

```python
class WorkflowGenerationError(Exception):
    def __init__(
        self,
        message: str,
        error_type: str = "GENERATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    )
```

### WorkflowExecutionError

워크플로우 실행 관련 예외입니다.

```python
class WorkflowExecutionError(Exception):
    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        error_type: str = "EXECUTION_ERROR",
        retryable: bool = True
    )
```

### LLMError

LLM 호출 관련 예외입니다.

```python
class LLMError(Exception):
    def __init__(
        self,
        message: str,
        error_type: str = "LLM_ERROR",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retryable: bool = True
    )
```

## 유틸리티 함수

### 워크플로우 유틸리티

```python
def validate_workflow_structure(workflow: WorkflowGraph) -> Tuple[bool, List[str]]
```

워크플로우 구조의 유효성을 검증합니다.

```python
def optimize_workflow_for_performance(workflow: WorkflowGraph) -> WorkflowGraph
```

성능을 위해 워크플로우를 최적화합니다.

```python
def estimate_workflow_cost(workflow: WorkflowGraph) -> float
```

워크플로우 실행 비용을 추정합니다.

### 패턴 유틸리티

```python
def extract_workflow_features(workflow: WorkflowGraph) -> Dict[str, Any]
```

워크플로우에서 특징을 추출합니다.

```python
def calculate_pattern_similarity(pattern1: WorkflowPattern, pattern2: WorkflowPattern) -> float
```

두 패턴 간의 유사도를 계산합니다.

```python
def merge_compatible_patterns(patterns: List[WorkflowPattern]) -> Optional[WorkflowPattern]
```

호환 가능한 패턴들을 병합합니다.

## 설정 및 상수

### 기본 설정

```python
DEFAULT_CONFIG = {
    "generation": {
        "default_mode": GenerationMode.BALANCED,
        "max_nodes_per_workflow": 50,
        "template_match_threshold": 0.8,
        "llm_fallback_enabled": True
    },
    "llm_client": {
        "default_timeout_seconds": 30,
        "max_retries": 3,
        "request_rate_limit": 60,  # per minute
        "cost_limit_per_request": 1.0  # USD
    },
    "pattern_learning": {
        "min_executions_for_learning": 3,
        "confidence_threshold": 0.7,
        "max_patterns_per_task_type": 20,
        "pattern_cleanup_interval_hours": 24
    },
    "performance": {
        "execution_timeout_seconds": 300,
        "memory_limit_mb": 1024,
        "parallel_execution_max_workers": 4
    }
}
```

### 환경 변수

필요한 환경 변수들:

```bash
# LLM API 키
OPENAI_API_KEY=sk-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
ANTHROPIC_API_KEY=sk-ant-...
LITELLM_API_KEY=...

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434

# 패턴 저장소 설정
PATTERN_STORE_PATH=./patterns.json
PATTERN_BACKUP_PATH=./patterns_backup.json

# 로깅 설정
CODE_GENERATION_LOG_LEVEL=INFO
LLM_CLIENT_LOG_LEVEL=DEBUG
PATTERN_LEARNING_LOG_LEVEL=INFO
```

## 통합 예제

### 완전한 워크플로우 생성 및 실행

```python
from src.layers.code_generation import (
    WorkflowCodeGenerator, 
    GenerationMode,
    LLMClientFactory,
    WorkflowPatternStore
)

# 1. 시스템 초기화
pattern_store = WorkflowPatternStore("my_patterns.json")
llm_factory = LLMClientFactory()
generator = WorkflowCodeGenerator(
    llm_client_factory=llm_factory,
    pattern_store_path="my_patterns.json",
    enable_learning=True
)

# 2. 워크플로우 생성
workflow = generator.generate_workflow(
    task_description="""
    고객 리뷰 데이터를 분석하여 다음을 수행:
    1. 감정 분석 (긍정/부정/중립)
    2. 주요 키워드 추출
    3. 개선점 요약
    4. 고객 만족도 점수 계산
    """,
    mode=GenerationMode.BALANCED,
    task_type="sentiment_analysis",
    complexity_hint="medium",
    prompt_processing={
        "use_cot": True,
        "use_self_refine": True
    }
)

# 3. 워크플로우 검증
is_valid, errors = workflow.validate()
if not is_valid:
    print("워크플로우 검증 실패:", errors)
    exit(1)

# 4. 워크플로우 실행
initial_data = {
    "reviews": [
        "제품 품질이 우수하고 배송도 빨랐습니다!",
        "가격 대비 성능이 아쉽네요.",
        "고객 서비스가 친절했어요.",
        "배송 포장이 엉성했습니다."
    ]
}

try:
    result = generator.execute_workflow(
        workflow=workflow,
        initial_data=initial_data,
        learn_from_execution=True,
        max_retries=3
    )
    
    # 5. 결과 출력
    print("분석 결과:")
    print(f"전체 감정: {result['sentiment_summary']}")
    print(f"주요 키워드: {result['keywords']}")
    print(f"만족도 점수: {result['satisfaction_score']}/10")
    print(f"개선점: {result['improvement_suggestions']}")
    
except Exception as e:
    print(f"실행 중 오류: {e}")

# 6. 성능 통계 확인
stats = generator.get_statistics()
print(f"전체 생성 횟수: {stats['generation_stats']['total_generations']}")
print(f"성공률: {stats['generation_stats']['successful_generations'] / stats['generation_stats']['total_generations'] * 100:.1f}%")
```

### 고급 패턴 학습 및 적용

```python
from src.layers.code_generation.patterns import PatternLearner, PatternMatcher

# 패턴 학습 설정
learner = PatternLearner(pattern_store)
matcher = PatternMatcher(pattern_store)

# 다양한 시나리오로 패턴 학습
scenarios = [
    {
        "task": "이메일 자동 분류",
        "type": "classification",
        "data": {"emails": ["고객 불만", "제품 문의", "배송 관련"]}
    },
    {
        "task": "문서 요약",
        "type": "summarization", 
        "data": {"document": "긴 문서 내용..."}
    }
]

for scenario in scenarios:
    # 워크플로우 생성 및 실행
    workflow = generator.generate_workflow(
        task_description=scenario["task"],
        task_type=scenario["type"]
    )
    
    result = generator.execute_workflow(workflow, scenario["data"])
    
    # 성공한 경우 패턴 학습
    if result.get("success", False):
        learner.learn_from_workflow(
            workflow=workflow,
            execution_result=result,
            task_description=scenario["task"],
            task_type=scenario["type"]
        )

# 학습된 패턴으로 새 워크플로우 생성
new_task = "고객 피드백 분석"
matching_patterns = matcher.find_best_patterns(new_task)

if matching_patterns:
    print(f"매칭된 패턴: {matching_patterns[0].name}")
    optimized_workflow = generator.generate_workflow(
        task_description=new_task,
        mode=GenerationMode.LEARNING,
        use_learned_patterns=True
    )
```

---

**작성일**: 2025년 1월  
**버전**: 1.0  
**담당**: API Reference Documentation Team