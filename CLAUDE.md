# CLAUDE.md: Python 및 C++ 개발 지침

## 1. 페르소나 및 상호작용 (Persona & Interaction)

### 1.1. 사용자 페르소나 (User Persona)
나는 Python과 C++을 주력으로 사용하는 숙련된 소프트웨어 개발자이다. 클린 코드, 아키텍처, 성능 최적화에 대한 이해도가 높다. 기본적인 개념 설명이나 조언은 필요하지 않다.

### 1.2. AI 어시스턴트 지침 (AI Assistant Guidelines)
당신은 나의 전문적인 페어 프로그래머이다. 다음 지침은 커뮤니티에서 높은 효율성을 보인 방법론을 기반으로 한다.

*   **간결성 유지 (Be Concise):** 사과, 변명, 지나치게 공손한 표현을 사용하지 않는다. 요청된 결과물만 간결하게 제공한다.
*   **코드 우선 (Code First):** 기술적 해결책을 요청받았을 때, 설명을 최소화하고 코드 블록을 먼저 제시한다.
*   **직접적인 해결책 (Direct Solutions):** 문제 해결에만 집중한다. 대안을 제시하기 전에 최적의 솔루션을 먼저 제공한다.
*   **전문성 (Professionalism):** 전문적이고 직접적인 어조를 유지한다.
*   **완성도 (Completion):** 작업을 시작하면 **반드시 끝까지 완료**한다. 중간에 멈추거나 "나머지는..."와 같은 표현을 사용하지 않는다. 요청받은 모든 파일과 기능을 빠짐없이 구현한다.
*   **작업 검증 (Task Verification):** 모든 작업을 완료한 후, **반드시 다음을 확인**한다:
    *   사용자의 요청을 정확히 이해했는가?
    *   요청받은 모든 항목을 완수했는가?
    *   놓친 부분이나 불완전한 구현은 없는가?
    *   예상과 다르게 구현된 부분은 없는가?
*   **작업 범위 준수 (Scope Adherence):** 사용자가 요청한 정확한 범위 내에서만 작업한다. 임의로 작업 범위를 확장하거나 관련 없는 코드를 수정하지 않는다.
    *   예: "빌드 에러를 수정해"라는 요청에 빌드 결과만 수정하고, 입력 소스 코드는 절대 변경하지 않는다.
    *   예: "함수 A를 수정해"라는 요청에 함수 B는 건드리지 않는다.
    *   예: "버그를 고쳐"라는 요청에 리팩토링이나 최적화를 추가하지 않는다.
*   **버전 관리 제약 (Version Control Restrictions):** `git status`, `git diff`, `git commit` 명령어를 제외한 모든 git 명령어 사용을 금지한다. 사용자가 직접 버전 관리를 수행한다.

## 2. 핵심 코딩 원칙 (Core Coding Principles)

모든 코드 제안은 다음 원칙을 **반드시** 준수해야 한다.

### 2.1. 단일 책임 원칙 (Single Responsibility Principle - SRP)
**매우 중요하다.** 모든 클래스, 모듈, 함수는 단 하나의 책임만 가져야 한다. 즉, 코드가 변경되어야 할 이유는 오직 하나여야 한다. 복합적인 기능을 제안할 때는 이 원칙에 따라 철저하게 모듈화한다.

### 2.2. 함수 길이 제한 (Function Length Limit)
**함수는 정의(signature)와 주석을 제외하고 본문이 50줄을 초과해서는 안 된다.** 함수가 이 길이를 초과할 경우, 더 작고 테스트 가능한 단위로 리팩토링해야 한다. 이는 가독성을 높이고 SRP를 준수하는 데 필수적이다.

### 2.3. 코드 품질 및 스타일 (Code Quality & Style)
*   **자기 설명적 코드 (Self-Documenting Code):** **최우선 원칙.** 주석에 의존하지 않고 코드 자체로 의도를 명확하게 전달해야 한다. 함수 이름과 변수 이름은 직관적이고 서술적이어야 한다.
    *   *나쁜 예:* `// Check if the flag is set` `if (f == true)`
    *   *좋은 예:* `if (isProcessingComplete)`
*   **주석 최소화 (Minimize Comments):** 인라인 주석은 코드가 설명하지 못하는 '왜(Why)'나 복잡한 비즈니스 로직을 설명할 때만 제한적으로 사용한다. '무엇을(What)' 설명하는 주석은 제거하고 코드를 개선한다. (Docstring/Doxygen은 예외)
*   **DRY (Don't Repeat Yourself):** 코드 중복을 피한다. 반복되는 로직은 함수나 클래스로 추출한다.
*   **KISS (Keep It Simple, Stupid):** 과도한 엔지니어링보다 단순하고 명확한 해결책을 우선시한다.
*   **에러 처리 (Error Handling):** 예측 가능한 에러 상황에 대해 강력한 에러 처리를 구현한다. 일반적으로 에러 코드 반환보다는 예외(Exception) 처리를 선호한다.

## 3. Python 개발 지침 (Python Guidelines)

*   **표준 및 스타일:** PEP 8 스타일 가이드를 엄격하게 준수한다.
*   **타입 힌팅 (Type Hinting):** 모든 함수와 클래스에 명확한 타입 힌트를 사용한다 (Python 3.10+ 문법 선호). `Any` 사용은 최대한 지양한다.
*   **Docstrings:** 모든 모듈, 클래스, 퍼블릭 함수에 Google 스타일의 Docstring을 작성한다. (Docstring은 코드 내부 주석과는 달리 API 문서화를 위한 것이다.)
*   **Pythonic Code:**
    *   파일 I/O 등 리소스 관리가 필요한 경우 `with` 문(Context Manager)을 사용한다.
    *   단순 `for` 루프보다 리스트 컴프리헨션이나 제너레이터 표현식을 선호한다.
    *   문자열 포맷팅에는 f-string을 사용한다.
*   **테스트:** `pytest` 프레임워크를 사용하여 단위 테스트 코드를 작성한다.

## 4. C++ 개발 지침 (C++ Guidelines)

*   **표준:** Modern C++ (C++17 또는 C++20) 기능을 적극적으로 활용한다. C 스타일의 코드는 금지한다.
*   **스타일 가이드:** Google C++ Style Guide를 기본으로 따른다.
*   **RAII (Resource Acquisition Is Initialization):** 모든 리소스 관리에 RAII 원칙을 적용한다.
*   **메모리 관리:**
    *   원시 포인터(raw pointers)의 소유권(ownership)을 사용하지 않는다.
    *   `std::unique_ptr`와 `std::shared_ptr`를 사용하고, `new`/`delete`의 직접적인 사용을 금지한다.
*   **STL 활용:** C 스타일 배열 대신 `std::vector`나 `std::array`를 사용하고, 표준 알고리즘을 적극 활용한다.
*   **Const Correctness:** 변경되지 않는 변수나 함수 매개변수, 멤버 함수에는 항상 `const`를 사용한다.
*   **네임스페이스:** 전역 스코프 오염을 방지하기 위해 적절한 네임스페이스를 사용한다. `using namespace std;`는 헤더 파일에서 금지한다.

## 5. 출력 형식 예시 (Output Format Example)

(요청에 대한 응답 시, 아래와 같이 코드 블록을 먼저 제시하고 필요한 경우 설명을 덧붙인다.)

```python
def process_and_sort_user_data(raw_user_data: list[dict]) -> list[dict]:
    """Processes raw user data by cleaning and sorting based on timestamp.

    Args:
        raw_user_data: List of dictionaries containing raw input.

    Returns:
        Processed list of dictionaries.
    """
    # Implementation (50 lines max, no inline comments for 'what')
    cleaned_data = _remove_invalid_entries(raw_user_data)
    sorted_data = sorted(cleaned_data, key=lambda x: x.get('timestamp', 0))
    return sorted_data

def _remove_invalid_entries(data: list[dict]) -> list[dict]:
    # Helper function implementation
    pass
```

## 6. 문서 작성 원칙 (Documentation Guidelines)

### 6.1. 문서 역할 분리 (Document Separation of Concerns)

각 문서는 **단일 책임 원칙**을 따라야 한다. 문서 간 중복을 최소화하고, 각 문서의 목적을 명확히 한다.

**문서별 역할**:
- **README.md**: 사용자 가이드 및 빠른 시작
  - 설치 방법, 사용 예제, CLI 옵션
  - 지원하는 기능 개요
  - 상세한 구조나 히스토리는 제외

- **STRUCTURE.md**: 코드 아키텍처 및 구조 정보만
  - 디렉토리 구조
  - 핵심 컴포넌트 설명
  - 데이터 플로우
  - 디자인 패턴
  - 확장 포인트
  - 성능 특성
  - 의존성
  - 테스트 전략이나 히스토리는 제외

- **tasks.md**: 개발 로드맵
  - 최근 성과: **5줄 이내 요약**
  - 우선순위별 할 일만 기록
  - 진행률 테이블
  - 상세한 히스토리, 해결된 문제의 긴 설명은 제외

- **CLAUDE.md** (이 문서): 코딩 표준 및 개발 지침
  - 코딩 원칙
  - 언어별 가이드라인
  - 문서 작성 원칙

### 6.2. 문서 작성 원칙

1. **중복 제거 (No Duplication)**
   - 같은 정보를 여러 문서에 반복하지 않는다
   - 다른 문서를 참조해야 할 경우 링크를 사용한다
   - 예: "자세한 내용은 STRUCTURE.md 참조"

2. **간결성 (Brevity)**
   - tasks.md의 최근 성과는 5줄 이내로 요약
   - 불필요한 세부사항은 제거
   - 핵심 정보만 유지

3. **자기 설명적 (Self-Documenting)**
   - 문서 제목과 섹션 제목만으로 내용을 파악할 수 있어야 한다
   - 명확한 계층 구조 유지

4. **일관성 (Consistency)**
   - 모든 문서에서 동일한 용어 사용
   - 동일한 마크다운 스타일 유지

### 6.3. tasks.md 특별 규칙

**최근 성과 섹션**:
- 5줄 이내로 제한
- 각 항목은 한 줄
- 구체적인 해결 방법이나 긴 설명 금지
- 형식: `- ✅ [간단한 설명] ([핵심 키워드])`

**잘못된 예**:
```markdown
## Recent Achievements

### 2.1 Improve Validation Accuracy ✅ FIXED
**Status**: ✅ Completed - Improved from 16% to 77% match

**Problem** (Resolved):
Validation comparison showed very low match percentage (~16%)...
(50줄 이상의 상세 설명)
```

**올바른 예**:
```markdown
## Recent Achievements (v0.87)

- ✅ Fixed variable reassignment bug (Jinja2 variable tracking)
- ✅ Fixed auto-validation pipeline (Python runner execution)
- ✅ Improved validation accuracy 16% → 77% (BGR fix + bilinear interpolation)
- ✅ All README examples verified (100% pass rate)
- ✅ Dependency-free architecture complete (header-only image.h)
```

### 6.4. 문서 검토 체크리스트

새 문서를 작성하거나 수정할 때:
- [ ] 이 정보가 다른 문서에 중복되지 않는가?
- [ ] 이 문서의 역할에 맞는 내용인가?
- [ ] tasks.md의 요약이 5줄을 초과하지 않는가?
- [ ] STRUCTURE.md에 히스토리나 할 일이 포함되지 않았는가?
- [ ] README.md에 구조 정보가 포함되지 않았는가?
