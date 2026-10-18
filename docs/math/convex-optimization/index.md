---
title: 볼록 최적화
tags:
  - math
  - optimization
  - convex-optimization
---

# 볼록 최적화

Stephen Boyd와 Lieven Vandenberghe의 *Convex Optimization* (Cambridge University Press, 2004)을
장 단위로 정리합니다. 이 책은 세 부분으로 나뉩니다. 1부는 볼록 집합, 볼록 함수, 쌍대성의 이론을 세웁니다.
2부는 그 이론으로 근사, 통계 추정, 기하 문제를 볼록 문제로 바꾸는 방법을 보여 줍니다.
3부는 그 문제들을 실제로 푸는 뉴턴법과 내부점법을 다룹니다.

각 장은 책과 Stanford EE364a 강의 슬라이드를 기준으로 **핵심 개념만 골라 수식으로 따라가는 방식**으로
구성했습니다. 정의와 정리를 먼저 세우고, 짧은 유도와 수치 예시로 그 의미를 확인합니다.
딥러닝과 연결되는 지점이 있으면 함께 적었습니다.

!!! info "원문과 자료"

    - 교재 PDF: <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf> (저자가 공개한 무료 판)
    - 강의 슬라이드 전체: <https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf>
    - 강의 페이지: <https://web.stanford.edu/class/ee364a/> (Stanford EE364a, Convex Optimization I)
    - 책 홈페이지: <https://web.stanford.edu/~boyd/cvxbook/> (연습 문제, 추가 연습 문제, 예제 코드)

## 목차

### 1부 이론

| 장 | 다루는 내용 | 교재 | 슬라이드 |
| --- | --- | --- | --- |
| [1. 서론](01-introduction.md) | 최적화 문제의 형식, 최소제곱과 선형 계획, 볼록 최적화가 풀기 쉬운 이유를 정리합니다. | p.1 | [p.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=2) |
| [2. 볼록 집합](02-convex-sets.md) | 대표적인 볼록 집합, 볼록성을 보존하는 연산, 일반화 부등식, 분리 초평면 정리를 다룹니다. | p.21 | [p.20](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=20) |
| [3. 볼록 함수](03-convex-functions.md) | 1차·2차 조건, 볼록성 보존 연산, 켤레 함수, 준볼록과 로그 볼록을 다룹니다. | p.67 | [p.44](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=44) |
| [4. 볼록 최적화 문제](04-convex-problems.md) | 표준형, 최적성 조건, LP부터 SDP까지의 문제 계층, 기하 계획법, 다목적 최적화를 다룹니다. | p.127 | [p.87](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=87) |
| [5. 쌍대성](05-duality.md) | Lagrange 쌍대, 강쌍대성과 Slater 조건, KKT 조건, 민감도 분석, 대안 정리를 다룹니다. | p.215 | [p.154](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=154) |

### 2부 응용

| 장 | 다루는 내용 | 교재 | 슬라이드 |
| --- | --- | --- | --- |
| [6. 근사와 피팅](06-approximation.md) | 노름·페널티 근사, 정규화와 trade-off 곡선, 강건 근사를 다룹니다. | p.291 | [p.193](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=193) |
| [7. 통계적 추정](07-statistical-estimation.md) | 최대우도 추정, 잡음 모델과 페널티의 대응, 가설 검정, 실험 설계를 다룹니다. | p.351 | [p.218](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=218) |
| [8. 기하 문제](08-geometric-problems.md) | 극값 부피 타원체, 중심 문제, 선형 분류와 SVM, 배치 문제를 다룹니다. | p.397 | [p.243](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=243) |

### 3부 알고리즘

| 장 | 다루는 내용 | 교재 | 슬라이드 |
| --- | --- | --- | --- |
| [9. 제약 없는 최소화](09-unconstrained.md) | 강볼록성, 선 탐색, 경사 하강법과 뉴턴법의 수렴, self-concordant 함수를 다룹니다. | p.457 | [p.287](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=287) |
| [10. 등식 제약 최소화](10-equality-constrained.md) | KKT 시스템, feasible start와 infeasible start 뉴턴법, KKT 행렬 풀이를 다룹니다. | p.521 | [p.327](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=327) |
| [11. 내부점법](11-interior-point.md) | log barrier와 central path, barrier method, Phase I, 복잡도 분석, 원시-쌍대 방법을 다룹니다. | p.561 | [p.353](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf#page=353) |

교재 페이지는 책에 인쇄된 쪽 번호입니다. 슬라이드 링크는 전체 슬라이드 PDF의 해당 페이지를 엽니다.
부록 C(수치 선형대수)는 9~11장의 구현 절에서 필요한 부분만 인용합니다.

## 읽는 방법

1부는 순서대로 읽는 편이 좋습니다. 2장과 3장의 정의를 4장이 쓰고, 4장의 문제 형식을 5장이 씁니다.
2부의 세 장은 서로 독립적이므로 관심 있는 장부터 읽어도 됩니다. 3부는 9장의 뉴턴법을 10장과 11장이
차례로 확장하므로 순서대로 읽어야 합니다.

[신경망 기초](../../ai/d2l/index.md)의 [9. 최적화 알고리즘](../../ai/d2l/09-optimization.md)은 이 책의 9장과 주제가 겹칩니다.
그 문서는 비볼록 손실 위의 확률적 방법을 다루고, 이 책은 볼록 문제 위의 결정적 방법을 다룹니다.
