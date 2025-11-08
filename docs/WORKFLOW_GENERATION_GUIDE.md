# 워크플로우 생성 가이드

## 개요

Self-Evolving Agent Framework의 워크플로우 생성 시스템은 복잡한 작업을 자동화된 워크플로우로 변환하는 기능을 제공합니다. 이 가이드는 다양한 시나리오에서 워크플로우를 생성하고 실행하는 방법을 설명합니다.

## 기본 개념

### 워크플로우 구성 요소

1. **WorkflowNode**: 개별 작업 단위
2. **WorkflowGraph**: 노드들의 연결 관계
3. **WorkflowState**: 실행 상태 관리
4. **GenerationMode**: 생성 전략 선택

### 노드 타입

```python
from src.layers.code_generation.workflow import NodeType

# 사용 가능한 노드 타입들
NodeType.INPUT          # 사용자 입력
NodeType.LLM_CALL       # LLM 호출
NodeType.TOOL_USE       # 도구 사용
NodeType.DECISION       # 조건 분기
NodeType.LOOP           # 반복 처리
NodeType.OUTPUT         # 결과 출력
NodeType.AGGREGATION    # 데이터 집계
NodeType.VALIDATION     # 검증
```

## 기본 워크플로우 생성

### 1. 단순 Q&A 워크플로우

```python
from src.layers.code_generation import WorkflowCodeGenerator, GenerationMode

# 제너레이터 초기화
generator = WorkflowCodeGenerator()

# Q&A 워크플로우 생성
workflow = generator.generate_workflow(
    task_description="사용자 질문에 대해 AI가 답변하는 시스템",
    mode=GenerationMode.FAST,
    task_type="qa",
    complexity_hint="simple"
)

# 워크플로우 실행
results = generator.execute_workflow(
    workflow=workflow,
    initial_data={"question": "파이썬에서 리스트와 튜플의 차이는?"},
    learn_from_execution=True
)

print("답변:", results["final_output"]["answer"])
```

### 2. 웹 검색 + 분석 워크플로우

```python
# 복잡한 분석 워크플로우
workflow = generator.generate_workflow(
    task_description="""
    사용자 질문에 대해 다음 단계를 수행:
    1. 웹에서 관련 정보 검색
    2. 검색 결과 요약 
    3. 사용자 질문에 맞는 답변 생성
    """,
    mode=GenerationMode.BALANCED,
    task_type="research",
    complexity_hint="medium",
    required_tools=["web_search", "text_summarizer"]
)

# 실행
results = generator.execute_workflow(
    workflow=workflow,
    initial_data={"question": "2024년 AI 트렌드는?"}
)
```

## 수동 워크플로우 구성

### 노드별 생성 예시

```python
from src.layers.code_generation.workflow import WorkflowNode, WorkflowGraph, NodeType

# 1. 입력 노드
input_node = WorkflowNode(
    id="user_input",
    name="사용자 입력 받기",
    node_type=NodeType.INPUT,
    operation="receive_input",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"}
        }
    }
)

# 2. LLM 처리 노드
llm_node = WorkflowNode(
    id="llm_process", 
    name="LLM 답변 생성",
    node_type=NodeType.LLM_CALL,
    operation="generate_answer",
    model_name="gpt-4-turbo",
    prompt_template="""질문: {question}

위 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.
답변은 다음 형식으로 작성해주세요:

답변: [여기에 답변]
신뢰도: [1-10 점수]
관련 키워드: [키워드1, 키워드2, ...]""",
    output_parser="structured_response"
)

# 3. 검증 노드
validation_node = WorkflowNode(
    id="validate_answer",
    name="답변 품질 검증",
    node_type=NodeType.VALIDATION,
    operation="validate_response",
    validation_rules=[
        {"field": "신뢰도", "min_value": 7},
        {"field": "답변", "min_length": 50}
    ]
)

# 4. 출력 노드
output_node = WorkflowNode(
    id="format_output",
    name="최종 출력 포맷팅",
    node_type=NodeType.OUTPUT,
    operation="format_final_response",
    output_template={
        "answer": "{답변}",
        "confidence": "{신뢰도}",
        "keywords": "{관련 키워드}"
    }
)
```

### 워크플로우 그래프 구성

```python
# 워크플로우 그래프 생성
workflow = WorkflowGraph(
    id="qa_with_validation",
    name="검증이 포함된 Q&A"
)

# 노드 추가
workflow.add_node(input_node)
workflow.add_node(llm_node)
workflow.add_node(validation_node) 
workflow.add_node(output_node)

# 연결 관계 설정
workflow.connect("user_input", "llm_process", data_key="question")
workflow.connect("llm_process", "validate_answer", data_key="response")
workflow.connect("validate_answer", "format_output", data_key="validated_response")

# 조건부 연결 (검증 실패 시)
workflow.connect_conditional(
    from_node="validate_answer",
    to_node="llm_process",  # 다시 LLM 호출
    condition="validation_failed",
    data_key="feedback"
)
```

## 고급 워크플로우 패턴

### 1. 반복 처리 워크플로우

```python
# 데이터 배치 처리
batch_workflow = generator.generate_workflow(
    task_description="""
    여러 문서에 대해 다음을 수행:
    1. 각 문서 내용 읽기
    2. 핵심 내용 추출
    3. 카테고리 분류
    4. 결과 집계
    """,
    mode=GenerationMode.CREATIVE,
    task_type="batch_processing",
    workflow_patterns=["map_reduce", "parallel_processing"]
)

# 실행
results = generator.execute_workflow(
    workflow=batch_workflow,
    initial_data={
        "documents": [
            {"id": 1, "content": "문서1 내용..."},
            {"id": 2, "content": "문서2 내용..."},
            {"id": 3, "content": "문서3 내용..."}
        ]
    }
)
```

### 2. 결정 분기 워크플로우

```python
# 조건부 처리 워크플로우
decision_workflow = WorkflowGraph(id="conditional_analysis")

# 결정 노드 생성
decision_node = WorkflowNode(
    id="classify_request",
    name="요청 유형 분류",
    node_type=NodeType.DECISION,
    operation="classify_user_intent",
    decision_logic={
        "conditions": [
            {
                "if": "question_type == 'technical'",
                "then": "technical_analysis"
            },
            {
                "if": "question_type == 'general'", 
                "then": "general_qa"
            },
            {
                "default": "clarification_request"
            }
        ]
    }
)

# 각 분기별 처리 노드
technical_node = WorkflowNode(
    id="technical_analysis",
    name="기술적 분석",
    node_type=NodeType.LLM_CALL,
    model_name="gpt-4",
    prompt_template="기술적 질문에 대한 상세 분석: {question}"
)

general_node = WorkflowNode(
    id="general_qa",
    name="일반 질의응답",
    node_type=NodeType.LLM_CALL,
    model_name="gpt-3.5-turbo",
    prompt_template="일반적인 질문에 대한 답변: {question}"
)
```

### 3. 병렬 처리 워크플로우

```python
# 멀티 에이전트 병렬 처리
parallel_workflow = generator.generate_workflow(
    task_description="""
    사용자 질문에 대해 여러 관점에서 동시 분석:
    1. 기술적 관점 (Technical Agent)
    2. 비즈니스 관점 (Business Agent)  
    3. 사용자 경험 관점 (UX Agent)
    4. 결과 통합 및 종합
    """,
    mode=GenerationMode.CREATIVE,
    task_type="multi_agent_analysis",
    execution_strategy="parallel"
)

# 병렬 노드 정의
technical_agent = WorkflowNode(
    id="technical_agent",
    name="기술적 분석 에이전트",
    node_type=NodeType.LLM_CALL,
    model_name="gpt-4",
    prompt_template="기술적 관점에서 분석: {question}",
    parallel_group="analysis_agents"
)

business_agent = WorkflowNode(
    id="business_agent", 
    name="비즈니스 분석 에이전트",
    node_type=NodeType.LLM_CALL,
    model_name="claude-3-sonnet-20240229",
    prompt_template="비즈니스 관점에서 분석: {question}",
    parallel_group="analysis_agents"
)

# 결과 통합 노드
synthesis_node = WorkflowNode(
    id="synthesize_results",
    name="결과 통합",
    node_type=NodeType.AGGREGATION,
    operation="synthesize_multiple_perspectives",
    aggregation_strategy="weighted_merge",
    weights={
        "technical_agent": 0.4,
        "business_agent": 0.35,
        "ux_agent": 0.25
    }
)
```

## 생성 모드별 활용법

### FAST 모드 (템플릿 기반)

```python
# 빠른 처리가 필요한 간단한 작업
fast_workflow = generator.generate_workflow(
    task_description="간단한 텍스트 요약",
    mode=GenerationMode.FAST,
    task_type="summarization",
    template_hints=["text_processing", "single_step"]
)

# 특징: ~100ms 생성 시간, 높은 안정성
```

### BALANCED 모드 (하이브리드)

```python
# 일반적인 복합 작업
balanced_workflow = generator.generate_workflow(
    task_description="문서 분석 후 인사이트 도출",
    mode=GenerationMode.BALANCED,
    task_type="analysis",
    complexity_hint="medium"
)

# 특징: 템플릿 + LLM 조합, 적응적 전략
```

### CREATIVE 모드 (LLM 중심)

```python
# 새로운 유형의 복잡한 작업
creative_workflow = generator.generate_workflow(
    task_description="""
    창의적 콘텐츠 생성 파이프라인:
    1. 브레인스토밍
    2. 아이디어 평가 및 선별
    3. 상세 기획안 작성
    4. 실행 가능성 검토
    5. 최종 제안서 작성
    """,
    mode=GenerationMode.CREATIVE,
    task_type="creative_generation",
    allow_experimental_nodes=True
)

# 특징: 높은 창의성, 실험적 접근법
```

### LEARNING 모드 (패턴 기반)

```python
# 이전 학습 패턴 활용
learning_workflow = generator.generate_workflow(
    task_description="고객 지원 티켓 자동 분류",
    mode=GenerationMode.LEARNING,
    task_type="classification",
    use_learned_patterns=True,
    pattern_similarity_threshold=0.8
)

# 특징: 학습된 패턴 재사용, 높은 효율성
```

## 실시간 실행 및 모니터링

### 스트리밍 실행

```python
# 실시간 진행 상황 모니터링
def monitor_workflow_execution(workflow_id: str):
    for state in generator.executor.execute_workflow_streaming(workflow):
        print(f"현재 노드: {state.current_node}")
        print(f"진행률: {state.progress_percentage:.1f}%")
        print(f"상태: {state.status}")
        
        if state.status == "error":
            print(f"오류: {state.error_message}")
            break
        
        # 중간 결과 확인
        if state.intermediate_results:
            print(f"중간 결과: {state.intermediate_results}")

# 실행
monitor_workflow_execution("complex_analysis_workflow")
```

### 동적 워크플로우 수정

```python
# 실행 중 워크플로우 동적 수정
class AdaptiveWorkflowExecutor:
    def __init__(self, generator):
        self.generator = generator
    
    def execute_with_adaptation(self, workflow, initial_data):
        for state in self.generator.executor.execute_workflow_streaming(workflow):
            # 성능이 낮은 노드 감지
            if state.node_performance_score < 0.7:
                # 더 나은 모델로 교체
                improved_node = self.optimize_node(state.current_node)
                workflow.replace_node(state.current_node, improved_node)
            
            # 실행 계속
            if state.status == "completed":
                return state.final_results
    
    def optimize_node(self, node):
        if node.node_type == NodeType.LLM_CALL:
            # 더 강력한 모델로 업그레이드
            if "gpt-3.5" in node.model_name:
                node.model_name = "gpt-4-turbo"
            elif "claude-3-haiku" in node.model_name:
                node.model_name = "claude-3-sonnet-20240229"
        
        return node
```

## 워크플로우 최적화

### 성능 튜닝

```python
# 성능 기반 워크플로우 최적화
class WorkflowOptimizer:
    def __init__(self, generator):
        self.generator = generator
    
    def optimize_for_speed(self, workflow):
        """속도 최적화"""
        for node in workflow.nodes:
            if node.node_type == NodeType.LLM_CALL:
                # 빠른 모델로 교체
                if "gpt-4" in node.model_name:
                    node.model_name = "gpt-3.5-turbo"
                elif "claude-3-opus" in node.model_name:
                    node.model_name = "claude-3-haiku-20240307"
        
        return workflow
    
    def optimize_for_cost(self, workflow):
        """비용 최적화"""
        for node in workflow.nodes:
            if node.node_type == NodeType.LLM_CALL:
                # 경제적인 모델로 교체
                node.model_name = "gpt-3.5-turbo"
                # 토큰 수 제한
                if hasattr(node, 'max_tokens'):
                    node.max_tokens = min(node.max_tokens or 1000, 500)
        
        return workflow
    
    def optimize_for_quality(self, workflow):
        """품질 최적화"""
        for node in workflow.nodes:
            if node.node_type == NodeType.LLM_CALL:
                # 고성능 모델로 업그레이드
                if "gpt-3.5" in node.model_name:
                    node.model_name = "gpt-4-turbo"
        
        return workflow

# 사용법
optimizer = WorkflowOptimizer(generator)

# 용도별 최적화
speed_optimized = optimizer.optimize_for_speed(workflow.copy())
cost_optimized = optimizer.optimize_for_cost(workflow.copy())
quality_optimized = optimizer.optimize_for_quality(workflow.copy())
```

### A/B 테스팅

```python
# 워크플로우 A/B 테스팅
class WorkflowABTester:
    def __init__(self, generator):
        self.generator = generator
    
    def run_ab_test(self, workflow_a, workflow_b, test_cases, metrics=None):
        """두 워크플로우 버전을 비교 테스트"""
        metrics = metrics or ["accuracy", "speed", "cost"]
        
        results_a = self.run_test_suite(workflow_a, test_cases, "A")
        results_b = self.run_test_suite(workflow_b, test_cases, "B") 
        
        comparison = self.compare_results(results_a, results_b, metrics)
        
        return {
            "winner": comparison["winner"],
            "results_a": results_a,
            "results_b": results_b,
            "comparison": comparison
        }
    
    def run_test_suite(self, workflow, test_cases, version):
        results = []
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            
            result = self.generator.execute_workflow(
                workflow=workflow,
                initial_data=test_case["input"]
            )
            
            execution_time = time.time() - start_time
            
            results.append({
                "test_id": i,
                "version": version,
                "execution_time": execution_time,
                "output": result,
                "expected": test_case.get("expected"),
                "accuracy": self.calculate_accuracy(result, test_case.get("expected"))
            })
        
        return results

# 사용 예시
ab_tester = WorkflowABTester(generator)

test_cases = [
    {
        "input": {"question": "AI의 미래는?"},
        "expected": {"topic": "AI", "sentiment": "positive"}
    },
    # ... 더 많은 테스트 케이스
]

ab_results = ab_tester.run_ab_test(
    workflow_a=fast_workflow,
    workflow_b=quality_workflow, 
    test_cases=test_cases
)

print(f"승리자: {ab_results['winner']}")
```

## 트러블슈팅

### 일반적인 문제들

1. **워크플로우 검증 실패**
   ```python
   # 워크플로우 검증
   is_valid, errors = workflow.validate()
   if not is_valid:
       print("검증 오류:")
       for error in errors:
           print(f"- {error}")
   ```

2. **노드 실행 실패**
   ```python
   # 노드별 디버깅
   for node in workflow.nodes:
       try:
           result = generator.executor.execute_node(node, test_data)
           print(f"노드 {node.id}: 성공")
       except Exception as e:
           print(f"노드 {node.id}: 실패 - {e}")
   ```

3. **메모리 사용량 초과**
   ```python
   # 배치 크기 조절
   workflow.set_batch_size(max_parallel_nodes=3)
   
   # 메모리 사용량 모니터링
   import psutil
   
   def monitor_memory_usage():
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"메모리 사용량: {memory_mb:.1f} MB")
   ```

## 모범 사례

1. **명확한 태스크 정의**
   ```python
   # 좋은 예시
   task_description = """
   주어진 제품 리뷰 데이터에 대해:
   1. 감정 분석 (긍정/부정/중립)
   2. 주요 키워드 추출
   3. 카테고리별 점수 계산
   4. 개선사항 제안
   
   입력: JSON 형태의 리뷰 데이터
   출력: 구조화된 분석 보고서
   """
   ```

2. **적절한 모드 선택**
   ```python
   # 태스크 복잡도에 따른 모드 선택
   if task_complexity == "simple":
       mode = GenerationMode.FAST
   elif task_complexity == "medium":
       mode = GenerationMode.BALANCED
   else:
       mode = GenerationMode.CREATIVE
   ```

3. **에러 핸들링**
   ```python
   # 견고한 워크플로우 실행
   try:
       result = generator.execute_workflow(workflow, data)
   except WorkflowExecutionError as e:
       # 대체 워크플로우 시도
       fallback_workflow = generator.generate_workflow(
           task_description=simplified_task,
           mode=GenerationMode.FAST
       )
       result = generator.execute_workflow(fallback_workflow, data)
   ```

---

**작성일**: 2025년 1월  
**버전**: 1.0  
**담당**: Workflow Generation Implementation Team