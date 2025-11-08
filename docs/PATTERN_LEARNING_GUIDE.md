# 패턴 학습 시스템 가이드

## 개요

Self-Evolving Agent Framework의 패턴 학습 시스템은 성공한 워크플로우 패턴을 자동으로 학습하고 재사용하여 시스템 성능을 지속적으로 개선하는 핵심 기능입니다. 이 시스템을 통해 AI 에이전트는 경험에서 학습하고 점진적으로 더 나은 결과를 생성할 수 있습니다.

## 핵심 구성요소

### 1. WorkflowPatternStore
패턴 저장 및 검색을 담당하는 중앙 저장소

### 2. PatternLearner
실행 결과를 분석하여 새로운 패턴을 추출하는 학습 엔진

### 3. PatternMatcher
새로운 태스크에 적용할 수 있는 기존 패턴을 찾는 매칭 시스템

### 4. PerformanceTracker
패턴의 성능을 추적하고 최적화하는 모니터링 시스템

## 패턴 구조

### 기본 패턴 스키마

```python
from src.layers.code_generation.patterns import WorkflowPattern

pattern = WorkflowPattern(
    id="qa_with_validation_v1",
    name="검증이 포함된 Q&A 패턴",
    task_type="qa",
    complexity_level="medium",
    
    # 패턴 구조
    node_sequence=[
        {"type": "INPUT", "operation": "receive_question"},
        {"type": "LLM_CALL", "operation": "generate_answer", "model": "gpt-4-turbo"},
        {"type": "VALIDATION", "operation": "validate_answer"},
        {"type": "OUTPUT", "operation": "format_response"}
    ],
    
    # 연결 관계
    connections=[
        {"from": "receive_question", "to": "generate_answer", "data": "question"},
        {"from": "generate_answer", "to": "validate_answer", "data": "answer"},
        {"from": "validate_answer", "to": "format_response", "data": "validated_answer"}
    ],
    
    # 성능 메트릭
    performance_metrics={
        "success_rate": 0.95,
        "avg_execution_time": 2.3,
        "avg_cost": 0.05,
        "user_satisfaction": 4.2
    },
    
    # 적용 조건
    applicable_conditions=[
        {"task_type": "qa"},
        {"complexity": ["simple", "medium"]},
        {"quality_requirement": "high"}
    ],
    
    # 학습 메타데이터
    learning_metadata={
        "learned_from_executions": 47,
        "last_updated": "2025-01-15T10:30:00Z",
        "confidence_score": 0.92,
        "usage_count": 156
    }
)
```

## 패턴 학습 프로세스

### 1. 자동 패턴 학습

```python
from src.layers.code_generation.patterns import PatternLearner, WorkflowPatternStore

# 패턴 저장소 초기화
pattern_store = WorkflowPatternStore("patterns.json")
learner = PatternLearner(pattern_store)

# 워크플로우 실행 후 자동 학습
def execute_and_learn(generator, task_description, initial_data):
    # 1. 워크플로우 생성 및 실행
    workflow = generator.generate_workflow(
        task_description=task_description,
        mode=GenerationMode.BALANCED
    )
    
    start_time = time.time()
    result = generator.execute_workflow(workflow, initial_data)
    execution_time = time.time() - start_time
    
    # 2. 성능 평가
    performance = evaluate_result(result, initial_data)
    
    # 3. 성공한 경우 패턴 학습
    if performance["success"] and performance["quality_score"] > 8.0:
        learner.learn_from_workflow(
            workflow=workflow,
            execution_result={
                "success": True,
                "performance_score": performance["quality_score"],
                "execution_time_ms": execution_time * 1000,
                "cost_usd": estimate_cost(workflow, result),
                "user_feedback": performance.get("user_rating", 0)
            },
            task_description=task_description,
            task_type=infer_task_type(task_description),
            context_features=extract_context_features(initial_data)
        )
    
    return result

# 사용 예시
result = execute_and_learn(
    generator,
    "고객 리뷰 감정 분석 및 요약",
    {"reviews": ["좋은 제품입니다!", "배송이 늦었어요.", "품질이 우수해요."]}
)
```

### 2. 패턴 추출 세부 과정

```python
class AdvancedPatternLearner:
    def __init__(self, pattern_store):
        self.pattern_store = pattern_store
        self.feature_extractor = WorkflowFeatureExtractor()
        
    def learn_from_execution_batch(self, execution_logs):
        """배치 실행 로그에서 패턴 학습"""
        
        # 1. 성공한 워크플로우들 필터링
        successful_workflows = [
            log for log in execution_logs 
            if log['success'] and log['performance_score'] > 7.5
        ]
        
        # 2. 유사한 태스크들을 그룹핑
        task_groups = self.group_similar_tasks(successful_workflows)
        
        # 3. 각 그룹에서 공통 패턴 추출
        for group_key, workflows in task_groups.items():
            if len(workflows) >= 3:  # 최소 3개 이상의 성공 사례
                pattern = self.extract_common_pattern(workflows)
                
                if pattern.confidence_score > 0.8:
                    self.pattern_store.add_pattern(pattern)
    
    def extract_common_pattern(self, workflows):
        """여러 워크플로우에서 공통 패턴 추출"""
        
        # 노드 시퀀스 패턴 분석
        common_sequences = self.find_common_node_sequences(workflows)
        
        # 연결 패턴 분석
        common_connections = self.find_common_connections(workflows)
        
        # 파라미터 패턴 분석
        optimal_parameters = self.find_optimal_parameters(workflows)
        
        # 성능 통계 계산
        performance_stats = self.calculate_performance_stats(workflows)
        
        return WorkflowPattern(
            id=generate_pattern_id(common_sequences),
            node_sequence=common_sequences,
            connections=common_connections,
            parameters=optimal_parameters,
            performance_metrics=performance_stats,
            confidence_score=self.calculate_confidence(workflows)
        )
    
    def find_common_node_sequences(self, workflows):
        """공통 노드 시퀀스 찾기"""
        sequences = []
        
        for workflow in workflows:
            sequence = []
            for node in workflow['workflow'].get_execution_order():
                sequence.append({
                    'type': node.node_type.value,
                    'operation': node.operation,
                    'model': getattr(node, 'model_name', None)
                })
            sequences.append(sequence)
        
        # 최빈값 기반 공통 시퀀스 추출
        common_sequence = self.extract_most_common_sequence(sequences)
        
        return common_sequence
```

## 패턴 매칭 및 적용

### 1. 유사도 기반 매칭

```python
from src.layers.code_generation.patterns import PatternMatcher

class SmartPatternMatcher:
    def __init__(self, pattern_store):
        self.pattern_store = pattern_store
        
    def find_best_patterns(self, task_description, task_context=None, top_k=3):
        """주어진 태스크에 가장 적합한 패턴들을 찾음"""
        
        # 1. 태스크 특징 추출
        task_features = self.extract_task_features(task_description, task_context)
        
        # 2. 모든 패턴과 유사도 계산
        pattern_scores = []
        
        for pattern in self.pattern_store.get_all_patterns():
            similarity = self.calculate_similarity(task_features, pattern)
            
            if similarity > 0.6:  # 최소 임계값
                pattern_scores.append((pattern, similarity))
        
        # 3. 점수순 정렬
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in pattern_scores[:top_k]]
    
    def calculate_similarity(self, task_features, pattern):
        """태스크 특징과 패턴 간의 유사도 계산"""
        
        similarity_components = {
            'task_type': self.task_type_similarity(
                task_features['task_type'], 
                pattern.task_type
            ),
            'complexity': self.complexity_similarity(
                task_features['complexity'],
                pattern.complexity_level
            ),
            'keywords': self.keyword_similarity(
                task_features['keywords'],
                pattern.keywords
            ),
            'structure': self.structure_similarity(
                task_features['expected_structure'],
                pattern.node_sequence
            )
        }
        
        # 가중치 적용
        weights = {
            'task_type': 0.4,
            'complexity': 0.2,
            'keywords': 0.25,
            'structure': 0.15
        }
        
        total_similarity = sum(
            similarity_components[key] * weights[key]
            for key in similarity_components
        )
        
        return total_similarity
    
    def keyword_similarity(self, task_keywords, pattern_keywords):
        """키워드 기반 유사도"""
        if not task_keywords or not pattern_keywords:
            return 0.0
            
        task_set = set(task_keywords)
        pattern_set = set(pattern_keywords)
        
        intersection = len(task_set & pattern_set)
        union = len(task_set | pattern_set)
        
        return intersection / union if union > 0 else 0.0
```

### 2. 패턴 기반 워크플로우 생성

```python
class PatternBasedGenerator:
    def __init__(self, pattern_store, llm_client_factory):
        self.pattern_store = pattern_store
        self.pattern_matcher = SmartPatternMatcher(pattern_store)
        self.llm_client_factory = llm_client_factory
    
    def generate_from_pattern(self, task_description, pattern=None):
        """패턴을 기반으로 워크플로우 생성"""
        
        if pattern is None:
            # 최적 패턴 자동 선택
            matching_patterns = self.pattern_matcher.find_best_patterns(task_description)
            
            if not matching_patterns:
                raise ValueError("적용 가능한 패턴이 없습니다")
            
            pattern = matching_patterns[0]  # 가장 유사한 패턴 선택
        
        # 패턴을 현재 태스크에 맞게 적응
        adapted_workflow = self.adapt_pattern_to_task(pattern, task_description)
        
        return adapted_workflow
    
    def adapt_pattern_to_task(self, pattern, task_description):
        """패턴을 현재 태스크에 맞게 조정"""
        
        workflow = WorkflowGraph(
            id=f"pattern_adapted_{pattern.id}_{int(time.time())}",
            name=f"Adapted {pattern.name}"
        )
        
        # 노드들을 패턴에 따라 생성
        node_mapping = {}
        
        for i, node_template in enumerate(pattern.node_sequence):
            node = self.create_node_from_template(node_template, task_description, i)
            workflow.add_node(node)
            node_mapping[i] = node.id
        
        # 연결 관계 설정
        for connection in pattern.connections:
            from_node = node_mapping[connection['from_index']]
            to_node = node_mapping[connection['to_index']]
            workflow.connect(from_node, to_node, connection.get('data_key', 'output'))
        
        return workflow
    
    def create_node_from_template(self, template, task_description, index):
        """템플릿에서 실제 노드 생성"""
        
        node_id = f"node_{index}_{template['operation']}"
        
        if template['type'] == 'LLM_CALL':
            # LLM 노드의 경우 프롬프트를 태스크에 맞게 조정
            prompt = self.adapt_prompt_to_task(
                template.get('prompt_template', ''),
                task_description
            )
            
            # 최적 모델 선택
            model = self.select_optimal_model(template, task_description)
            
            return WorkflowNode(
                id=node_id,
                name=template.get('name', f"LLM 처리 {index}"),
                node_type=NodeType.LLM_CALL,
                operation=template['operation'],
                model_name=model,
                prompt_template=prompt
            )
        
        elif template['type'] == 'VALIDATION':
            return WorkflowNode(
                id=node_id,
                name=template.get('name', f"검증 {index}"),
                node_type=NodeType.VALIDATION,
                operation=template['operation'],
                validation_rules=template.get('validation_rules', [])
            )
        
        # 다른 노드 타입들...
        
        return WorkflowNode(
            id=node_id,
            name=template.get('name', f"작업 {index}"),
            node_type=NodeType[template['type']],
            operation=template['operation']
        )
```

## 성능 추적 및 최적화

### 1. 패턴 성능 모니터링

```python
class PatternPerformanceTracker:
    def __init__(self, pattern_store):
        self.pattern_store = pattern_store
        self.performance_history = {}
    
    def track_pattern_usage(self, pattern_id, execution_result):
        """패턴 사용 성능 추적"""
        
        if pattern_id not in self.performance_history:
            self.performance_history[pattern_id] = []
        
        self.performance_history[pattern_id].append({
            'timestamp': datetime.now(),
            'success': execution_result.get('success', False),
            'performance_score': execution_result.get('performance_score', 0),
            'execution_time': execution_result.get('execution_time_ms', 0),
            'cost': execution_result.get('cost_usd', 0),
            'user_feedback': execution_result.get('user_feedback')
        })
        
        # 패턴 통계 업데이트
        self.update_pattern_statistics(pattern_id)
    
    def update_pattern_statistics(self, pattern_id):
        """패턴 통계 갱신"""
        
        history = self.performance_history[pattern_id]
        recent_history = history[-20:]  # 최근 20회 실행
        
        if len(recent_history) >= 5:
            # 성공률 계산
            success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
            
            # 평균 성능 점수
            avg_performance = sum(h['performance_score'] for h in recent_history) / len(recent_history)
            
            # 평균 실행 시간
            avg_time = sum(h['execution_time'] for h in recent_history) / len(recent_history)
            
            # 평균 비용
            avg_cost = sum(h['cost'] for h in recent_history) / len(recent_history)
            
            # 패턴 정보 업데이트
            pattern = self.pattern_store.get_pattern(pattern_id)
            if pattern:
                pattern.performance_metrics.update({
                    'success_rate': success_rate,
                    'avg_performance_score': avg_performance,
                    'avg_execution_time_ms': avg_time,
                    'avg_cost_usd': avg_cost,
                    'total_usage_count': len(history)
                })
                
                self.pattern_store.update_pattern(pattern)
    
    def get_pattern_rankings(self, task_type=None, metric='overall'):
        """패턴 성능 순위 조회"""
        
        patterns = self.pattern_store.find_patterns(task_type=task_type) if task_type else self.pattern_store.get_all_patterns()
        
        pattern_scores = []
        
        for pattern in patterns:
            if metric == 'overall':
                score = self.calculate_overall_score(pattern)
            elif metric == 'speed':
                score = 1000 / (pattern.performance_metrics.get('avg_execution_time_ms', 1000) + 1)
            elif metric == 'cost':
                score = 1 / (pattern.performance_metrics.get('avg_cost_usd', 0.01) + 0.001)
            elif metric == 'quality':
                score = pattern.performance_metrics.get('avg_performance_score', 0)
            else:
                score = pattern.performance_metrics.get(metric, 0)
            
            pattern_scores.append((pattern, score))
        
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        return pattern_scores
    
    def calculate_overall_score(self, pattern):
        """종합 점수 계산"""
        metrics = pattern.performance_metrics
        
        # 가중치
        weights = {
            'success_rate': 0.3,
            'performance_score': 0.25,
            'speed_score': 0.2,
            'cost_score': 0.15,
            'usage_popularity': 0.1
        }
        
        # 정규화된 점수들
        success_rate = metrics.get('success_rate', 0)
        performance_score = metrics.get('avg_performance_score', 0) / 10  # 0-1 범위로 정규화
        speed_score = 1000 / (metrics.get('avg_execution_time_ms', 1000) + 1)  # 빠를수록 높은 점수
        cost_score = 1 / (metrics.get('avg_cost_usd', 0.01) + 0.001)  # 저렴할수록 높은 점수
        usage_popularity = min(metrics.get('total_usage_count', 0) / 100, 1)  # 사용량 기반 인기도
        
        overall_score = (
            success_rate * weights['success_rate'] +
            performance_score * weights['performance_score'] +
            speed_score * weights['speed_score'] +
            cost_score * weights['cost_score'] +
            usage_popularity * weights['usage_popularity']
        )
        
        return overall_score
```

### 2. 패턴 진화 및 개선

```python
class PatternEvolutionEngine:
    def __init__(self, pattern_store, llm_client_factory):
        self.pattern_store = pattern_store
        self.llm_client_factory = llm_client_factory
        self.performance_tracker = PatternPerformanceTracker(pattern_store)
    
    def evolve_underperforming_patterns(self, min_performance_threshold=7.0):
        """성능이 낮은 패턴들을 개선"""
        
        all_patterns = self.pattern_store.get_all_patterns()
        
        for pattern in all_patterns:
            avg_performance = pattern.performance_metrics.get('avg_performance_score', 10)
            
            if avg_performance < min_performance_threshold:
                print(f"패턴 {pattern.id} 개선 시도 중... (현재 점수: {avg_performance:.1f})")
                
                # 개선 시도
                improved_pattern = self.improve_pattern(pattern)
                
                if improved_pattern:
                    # A/B 테스트로 검증
                    if self.validate_improved_pattern(pattern, improved_pattern):
                        # 기존 패턴을 개선된 버전으로 교체
                        self.pattern_store.replace_pattern(pattern.id, improved_pattern)
                        print(f"패턴 {pattern.id}가 성공적으로 개선되었습니다.")
    
    def improve_pattern(self, pattern):
        """개별 패턴 개선"""
        
        improvement_strategies = [
            self.improve_model_selection,
            self.improve_prompt_templates,
            self.improve_node_sequence,
            self.improve_validation_rules
        ]
        
        improved_pattern = pattern.copy()
        
        for strategy in improvement_strategies:
            try:
                improved_pattern = strategy(improved_pattern)
            except Exception as e:
                print(f"개선 전략 실행 중 오류: {e}")
                continue
        
        return improved_pattern
    
    def improve_model_selection(self, pattern):
        """모델 선택 최적화"""
        
        # 성능 기록 분석
        history = self.performance_tracker.performance_history.get(pattern.id, [])
        
        if len(history) < 10:
            return pattern  # 데이터 부족
        
        # 최근 성능이 좋지 않은 LLM 노드들 찾기
        for i, node_template in enumerate(pattern.node_sequence):
            if node_template.get('type') == 'LLM_CALL':
                current_model = node_template.get('model')
                
                # 더 나은 모델 제안
                better_model = self.suggest_better_model(
                    current_model, 
                    pattern.task_type,
                    history
                )
                
                if better_model != current_model:
                    pattern.node_sequence[i]['model'] = better_model
        
        return pattern
    
    def suggest_better_model(self, current_model, task_type, performance_history):
        """더 나은 모델 제안"""
        
        # 태스크 타입별 모델 성능 데이터
        model_performance = {
            'qa': {
                'gpt-3.5-turbo': {'speed': 9, 'cost': 9, 'quality': 7},
                'gpt-4-turbo': {'speed': 7, 'cost': 6, 'quality': 9},
                'claude-3-haiku-20240307': {'speed': 8, 'cost': 8, 'quality': 8}
            },
            'code_generation': {
                'gpt-4': {'speed': 5, 'cost': 4, 'quality': 10},
                'claude-3-sonnet-20240229': {'speed': 7, 'cost': 6, 'quality': 9}
            },
            'analysis': {
                'gpt-4': {'speed': 5, 'cost': 4, 'quality': 10},
                'claude-3-opus-20240229': {'speed': 4, 'cost': 3, 'quality': 10}
            }
        }
        
        current_performance = performance_history[-5:]  # 최근 5회
        avg_score = sum(h['performance_score'] for h in current_performance) / len(current_performance)
        
        if avg_score < 7.0:
            # 품질 우선 모델 선택
            candidates = model_performance.get(task_type, {})
            best_model = max(candidates.keys(), 
                           key=lambda m: candidates[m]['quality'], 
                           default=current_model)
            return best_model
        
        return current_model
    
    def validate_improved_pattern(self, original_pattern, improved_pattern):
        """개선된 패턴을 검증"""
        
        # 간단한 A/B 테스트 실행
        test_cases = self.generate_test_cases(original_pattern.task_type)
        
        original_scores = []
        improved_scores = []
        
        for test_case in test_cases[:5]:  # 5개 테스트 케이스
            try:
                # 원본 패턴 테스트
                orig_result = self.execute_pattern_test(original_pattern, test_case)
                original_scores.append(orig_result['performance_score'])
                
                # 개선된 패턴 테스트
                improved_result = self.execute_pattern_test(improved_pattern, test_case)
                improved_scores.append(improved_result['performance_score'])
                
            except Exception as e:
                print(f"패턴 테스트 중 오류: {e}")
                continue
        
        if len(original_scores) >= 3 and len(improved_scores) >= 3:
            original_avg = sum(original_scores) / len(original_scores)
            improved_avg = sum(improved_scores) / len(improved_scores)
            
            # 개선된 패턴이 유의미하게 더 좋은 성능을 보이는지 확인
            return improved_avg > original_avg * 1.1  # 10% 이상 개선
        
        return False
```

## 실전 사용 예시

### 1. 패턴 기반 워크플로우 생성

```python
from src.layers.code_generation import WorkflowCodeGenerator, GenerationMode
from src.layers.code_generation.patterns import WorkflowPatternStore, PatternLearner

# 시스템 초기화
pattern_store = WorkflowPatternStore("learned_patterns.json")
generator = WorkflowCodeGenerator(pattern_store_path="learned_patterns.json")

# 학습 모드로 워크플로우 생성 (기존 패턴 활용)
workflow = generator.generate_workflow(
    task_description="고객 리뷰 데이터 감정 분석 및 인사이트 도출",
    mode=GenerationMode.LEARNING,
    task_type="sentiment_analysis",
    use_learned_patterns=True
)

print(f"사용된 패턴: {workflow.metadata.get('applied_pattern_id', 'None')}")

# 실행 및 성능 추적
result = generator.execute_workflow(
    workflow=workflow,
    initial_data={"reviews": ["제품이 훌륭해요!", "배송이 늦었네요", "가성비 좋습니다"]},
    learn_from_execution=True
)
```

### 2. 패턴 성능 분석 및 최적화

```python
# 패턴 성능 리포트 생성
def generate_pattern_report(pattern_store):
    """패턴 성능 리포트 생성"""
    
    tracker = PatternPerformanceTracker(pattern_store)
    
    print("=== 패턴 성능 리포트 ===\n")
    
    # 태스크 타입별 최고 성능 패턴
    task_types = ["qa", "analysis", "code_generation", "sentiment_analysis"]
    
    for task_type in task_types:
        print(f"[{task_type.upper()}] 최고 성능 패턴:")
        
        rankings = tracker.get_pattern_rankings(task_type=task_type, metric='overall')
        
        for i, (pattern, score) in enumerate(rankings[:3]):
            metrics = pattern.performance_metrics
            print(f"  {i+1}. {pattern.name}")
            print(f"     종합점수: {score:.2f}")
            print(f"     성공률: {metrics.get('success_rate', 0)*100:.1f}%")
            print(f"     평균 품질: {metrics.get('avg_performance_score', 0):.1f}/10")
            print(f"     사용횟수: {metrics.get('total_usage_count', 0)}")
            print()
    
    # 개선 필요 패턴 식별
    print("개선이 필요한 패턴들:")
    all_patterns = pattern_store.get_all_patterns()
    
    for pattern in all_patterns:
        avg_score = pattern.performance_metrics.get('avg_performance_score', 10)
        success_rate = pattern.performance_metrics.get('success_rate', 1)
        
        if avg_score < 7.0 or success_rate < 0.8:
            print(f"- {pattern.name}: 평균점수 {avg_score:.1f}, 성공률 {success_rate*100:.1f}%")

# 리포트 실행
generate_pattern_report(pattern_store)
```

### 3. 커스텀 패턴 수동 등록

```python
# 검증된 워크플로우 패턴을 수동으로 등록
def register_custom_pattern():
    """성공적인 워크플로우를 패턴으로 등록"""
    
    custom_pattern = WorkflowPattern(
        id="email_auto_response_v1",
        name="이메일 자동 응답 생성",
        task_type="email_generation",
        complexity_level="medium",
        
        node_sequence=[
            {
                "type": "INPUT",
                "operation": "receive_email",
                "name": "이메일 내용 분석"
            },
            {
                "type": "LLM_CALL", 
                "operation": "analyze_intent",
                "name": "의도 파악",
                "model": "gpt-4-turbo",
                "prompt_template": "다음 이메일의 의도를 분석해주세요: {email_content}"
            },
            {
                "type": "DECISION",
                "operation": "route_response_type",
                "name": "응답 유형 결정",
                "decision_logic": {
                    "conditions": [
                        {"if": "intent == 'complaint'", "then": "complaint_response"},
                        {"if": "intent == 'inquiry'", "then": "inquiry_response"},
                        {"default": "general_response"}
                    ]
                }
            },
            {
                "type": "LLM_CALL",
                "operation": "generate_response", 
                "name": "응답 생성",
                "model": "gpt-4-turbo",
                "prompt_template": "고객 {intent}에 대한 적절한 이메일 응답을 작성해주세요: {email_content}"
            }
        ],
        
        performance_metrics={
            "success_rate": 0.92,
            "avg_execution_time_ms": 3200,
            "avg_cost_usd": 0.08,
            "user_satisfaction": 4.3
        },
        
        applicable_conditions=[
            {"task_type": "email_generation"},
            {"domain": "customer_service"},
            {"language": "korean"}
        ]
    )
    
    pattern_store.add_pattern(custom_pattern)
    print(f"패턴 '{custom_pattern.name}' 등록 완료")

register_custom_pattern()
```

## 패턴 관리 도구

### 패턴 시각화 및 분석

```python
class PatternAnalyzer:
    def __init__(self, pattern_store):
        self.pattern_store = pattern_store
    
    def visualize_pattern_network(self):
        """패턴들 간의 관계를 시각화"""
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.Graph()
        
        patterns = self.pattern_store.get_all_patterns()
        
        # 노드 추가 (패턴들)
        for pattern in patterns:
            G.add_node(pattern.id, 
                      name=pattern.name,
                      task_type=pattern.task_type,
                      performance=pattern.performance_metrics.get('avg_performance_score', 0))
        
        # 엣지 추가 (유사한 패턴들)
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                similarity = self.calculate_pattern_similarity(pattern1, pattern2)
                if similarity > 0.7:
                    G.add_edge(pattern1.id, pattern2.id, weight=similarity)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # 성능에 따른 노드 색상
        node_colors = [G.nodes[node]['performance'] for node in G.nodes()]
        
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                cmap=plt.cm.RdYlGn)
        
        plt.title("패턴 관계 네트워크")
        plt.colorbar(label="성능 점수")
        plt.show()
    
    def export_pattern_summary(self, filename="pattern_summary.csv"):
        """패턴 요약 정보를 CSV로 내보내기"""
        import pandas as pd
        
        patterns = self.pattern_store.get_all_patterns()
        
        data = []
        for pattern in patterns:
            metrics = pattern.performance_metrics
            data.append({
                'ID': pattern.id,
                'Name': pattern.name,
                'TaskType': pattern.task_type,
                'Complexity': pattern.complexity_level,
                'SuccessRate': metrics.get('success_rate', 0),
                'AvgPerformance': metrics.get('avg_performance_score', 0),
                'AvgTime(ms)': metrics.get('avg_execution_time_ms', 0),
                'AvgCost($)': metrics.get('avg_cost_usd', 0),
                'UsageCount': metrics.get('total_usage_count', 0),
                'ConfidenceScore': pattern.learning_metadata.get('confidence_score', 0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"패턴 요약이 {filename}에 저장되었습니다.")

# 분석 도구 사용
analyzer = PatternAnalyzer(pattern_store)
analyzer.visualize_pattern_network()
analyzer.export_pattern_summary("my_patterns.csv")
```

## 트러블슈팅

### 일반적인 문제들

1. **패턴 매칭 실패**
   ```python
   # 매칭 임계값 조정
   matcher = PatternMatcher(pattern_store)
   matcher.similarity_threshold = 0.5  # 기본값: 0.6
   
   # 더 포괄적인 검색
   patterns = matcher.find_best_patterns(
       task_description,
       use_fuzzy_matching=True,
       include_partial_matches=True
   )
   ```

2. **패턴 성능 저하**
   ```python
   # 패턴 리프레시
   evolution_engine = PatternEvolutionEngine(pattern_store, llm_client_factory)
   evolution_engine.evolve_underperforming_patterns(min_performance_threshold=6.0)
   ```

3. **패턴 저장소 크기 증가**
   ```python
   # 오래되거나 성능이 낮은 패턴 정리
   def cleanup_patterns(pattern_store, max_patterns=100):
       patterns = pattern_store.get_all_patterns()
       
       if len(patterns) > max_patterns:
           # 성능순으로 정렬하여 상위 패턴만 유지
           tracker = PatternPerformanceTracker(pattern_store)
           rankings = tracker.get_pattern_rankings(metric='overall')
           
           # 하위 패턴들 제거
           for pattern, _ in rankings[max_patterns:]:
               pattern_store.remove_pattern(pattern.id)
   
   cleanup_patterns(pattern_store)
   ```

---

**작성일**: 2025년 1월  
**버전**: 1.0  
**담당**: Pattern Learning System Implementation Team