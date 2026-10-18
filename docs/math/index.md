---
title: 수학
tags:
  - math
---

# 수학

모델의 동작을 이해하는 데 필요한 수학을 주제별 교재 한 권씩으로 정리합니다.
개념을 나열하기보다, 딥러닝의 어느 지점에서 그 개념이 실제로 쓰이는지를 함께 적습니다.

이 섹션은 특정 모델과 무관하게 성립하는 이론을 다룹니다.
그 이론을 신경망에 적용하는 내용은 [AI](../ai/index.md) 섹션에 둡니다.
예를 들어 볼록 문제의 경사 하강법과 뉴턴법은 이 섹션의 [볼록 최적화](convex-optimization/index.md)에 있습니다.
비볼록 손실 위의 SGD와 Adam은 AI 섹션의 [최적화 알고리즘](../ai/d2l/09-optimization.md)에 있습니다.

## 주제

| 주제 | 교재 | 상태 | 정리 |
| --- | --- | --- | --- |
| 선형대수 | Strang, *Linear Algebra and Its Applications* | 1~8장, 부록 정리 | [바로 가기](linear-algebra/index.md) |
| 확률론 | Evans & Rosenthal, *Probability and Statistics* | 1~11장 정리 | [바로 가기](probability/index.md) |
| 볼록 최적화 | Boyd & Vandenberghe, *Convex Optimization* | 1~11장 정리 | [바로 가기](convex-optimization/index.md) |

## 새 주제를 추가하는 방법

1. 주제가 이 섹션에 속하는지 확인합니다. 신경망 구조나 학습 기법을 다루는 교재는 `docs/ai/` 에 둡니다.
2. `docs/math/` 아래에 주제 이름을 딴 디렉토리를 만들고 `index.md` 에 교재 개요와 목차를 적습니다.
3. 장별 문서는 `01-`, `02-` 처럼 번호를 앞에 붙입니다.
4. `mkdocs.yml` 의 `nav` 에 새 항목을 추가합니다.

## 수식 표기 규칙

- 인라인 수식은 `$...$` 로, 블록 수식은 `$$...$$` 로 작성합니다.
- 벡터는 소문자 $x$, 행렬은 대문자 $A$ 로 표기합니다.
- 차원을 처음 도입할 때에는 $A \in \mathbb{R}^{m \times n}$ 처럼 반드시 명시합니다.
