---
title: 양자 컴퓨팅
tags:
  - cs
  - quantum-computing
---

# 양자 컴퓨팅

Kaye, Laflamme, Mosca의 *An Introduction to Quantum Computing* (2007)을 10개 장으로 정리합니다.

양자역학의 중첩(superposition)과 얽힘(entanglement), 그리고 간섭(interference)을 계산 모형으로 정식화하고 알고리즘과 오류 정정까지 다룹니다. 책은 크게 세 부로 나뉩니다. 1~5장에서는 가역 계산, 디랙 표기(Dirac notation), 큐비트(qubit) 공준, 양자 회로와 보편 게이트(universal gate), 초고밀도 코딩과 텔레포테이션을 배웁니다. 6~8장에서는 위상 반전(phase kickback), 양자 푸리에 변환(QFT), 위상 추정(QPE), 쇼어(Shor) 알고리즘, 그로버(Grover) 진폭 증폭을 유도합니다. 9~10장에서는 계산 복잡도 계층(`BPP`, `BQP`), 질의 복잡도(query complexity) 하한 증명법, 양자 오류 정정 부호와 결함 허용(fault-tolerant) 문턱 정리(threshold theorem)를 다룹니다.

부록 A의 수학 도구는 따로 떼지 않고 관련 장에 통합했습니다. 군론과 유한체(A.7, A.8)는 2장과 3장에, 확률 부등식(A.1, A.3)은 6장에, 정수론 정리(A.2, A.4)는 7장에, 거리와 스펙트럼 노름(A.5, A.9)은 9장에 배치했습니다.

!!! info "원문과 참고 자료"
    - **주 교재**: Phillip Kaye, Raymond Laflamme, Michele Mosca, *An Introduction to Quantum Computing*, Oxford University Press, 2007 (ISBN 978-0-19-857049-3).
    - **선수 지식**: 복소수, 행렬 곱셈과 고윳값 분해, 기초 확률론, 점근 표기법($O, \Omega, \Theta$). 물리학 사전 지식은 1장과 3장에서 필요한 만큼만 도입합니다.

---

## 주요 기호와 게이트

본문 전체에서 아래 기호와 단일·다중 큐비트 게이트 표기를 따릅니다.

| 기호 | 정의 | 행렬 또는 의미 |
| --- | --- | --- |
| $\lvert 0 \rangle, \lvert 1 \rangle$ | 계산 기저(computational basis) | $\begin{pmatrix} 1 \\ 0 \end{pmatrix}, \ \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ |
| $\lvert + \rangle, \lvert - \rangle$ | 아다마르 기저 | $\frac{1}{\sqrt{2}}(\lvert 0 \rangle + \lvert 1 \rangle), \ \frac{1}{\sqrt{2}}(\lvert 0 \rangle - \lvert 1 \rangle)$ |
| $X, Y, Z$ | 파울리 게이트 | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \ \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \ \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |
| $H$ | 아다마르(Hadamard) 게이트 | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ |
| $S, T$ | 위상(phase) 및 $\pi/8$ 게이트 | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \ \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ |
| $\mathrm{CNOT}$ | 제어-NOT 게이트 | $\lvert a, b \rangle \mapsto \lvert a, b \oplus a \rangle$ |
| $\mathrm{SWAP}$ | 두 큐비트 교환 게이트 | $\lvert a, b \rangle \mapsto \lvert b, a \rangle$ |
| $\rho$ | 밀도 연산자(density operator) | $\sum_j p_j \lvert \psi_j \rangle \langle \psi_j \rvert$ |
| $D(\rho, \sigma)$ | 트레이스 거리(trace distance) | $\frac{1}{2}\mathrm{Tr}\lvert \rho - \sigma \rvert$ |
| $F(\rho, \sigma)$ | 충실도(fidelity) | $\mathrm{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}$ |

---

## 목차

### 1부. 기초, 회로 모델, 양자 통신

| 장 | 주제 | 핵심 내용과 대표 예시 |
| --- | --- | --- |
| [1. 서론과 배경](01-introduction.md) | 계산 모델과 양자 간섭 | 비트와 확률 비트, 토폴리 게이트와 무계산(uncomputation), 마흐-젠더 간섭계의 위상 소거 |
| [2. 선형대수와 디랙 표기](02-linear-algebra-dirac.md) | 복소 벡터 공간과 분해 정리 | 브라-켓 표기, 텐서곱, 스펙트럼 분해, 극분해·SVD, 슈미트 분해 ($2$-큐비트 $\mathrm{SVD}$ 수치 검증) |
| [3. 큐비트와 양자역학의 틀](03-qubits-framework.md) | 4가지 양자역학 공준 | 블로흐 구(Bloch sphere), 벨 상태(Bell state), 밀도 연산자, 축소 밀도 행렬, 크라우스 연산자 |
| [4. 양자 계산 모델과 회로](04-quantum-circuits.md) | 보편 게이트와 근사 | $Z\text{-}Y\text{-}Z$ 분해, 제어-$U$ 구성, 지연 측정 원리, 솔로베이-키타예프 정리, $\{H, T, \mathrm{CNOT}\}$ 보편성 |
| [5. 초고밀도 코딩과 양자 텔레포테이션](05-superdense-teleportation.md) | 얽힘 기반 통신과 계산 | 초고밀도 코딩($1$ 큐비트로 $2$ 고전 비트 전송), 텔레포테이션, 게이트 텔레포테이션의 $16$가지 파울리 전파 |

### 2부. 양자 알고리즘

| 장 | 주제 | 핵심 내용과 대표 예시 |
| --- | --- | --- |
| [6. 기초 양자 알고리즘](06-intro-algorithms.md) | 오라클 모델과 초기 알고리즘 | 위상 반전, 도이치-조사(Deutsch-Jozsa), 번스타인-바지라니, 사이먼(Simon) 알고리즘, 체르노프 증폭 |
| [7. 초다항 가속 알고리즘](07-superpolynomial-algorithms.md) | 푸리에 변환과 숨은 부분군 | 양자 푸리에 변환(QFT), 위상 추정(QPE) 오차 분포($\omega = 5.3/16$), 쇼어 인수분해($N=15$), 아벨 HSP |
| [8. 진폭 증폭 알고리즘](08-amplitude-amplification.md) | 그로버 탐색과 양자 계수 | 반전 연산자, $2$차원 회전 기하학, 일반 진폭 증폭, 양자 계수(Quantum Counting)와 진폭 추정(QAE) |

### 3부. 복잡도 하한과 오류 정정

| 장 | 주제 | 핵심 내용과 대표 예시 |
| --- | --- | --- |
| [9. 양자 계산 복잡도와 하한](09-complexity-lower-bounds.md) | 복잡도 계층과 3가지 하한 기법 | `P`, `BPP`, `BQP` 관계, 헬스트룀 한계, 하이브리드 논증, 다항식 방법, 양자 대적자(adversary) 방법 |
| [10. 양자 오류 정정](10-error-correction.md) | 안정자 부호와 결함 허용 | 비트·위상 플립 부호, 쇼어 $9$-큐비트 부호, 안정자(stabilizer)와 CSS 부호, 횡단 게이트와 문턱 정리 |

---

## 읽는 방법

1. 선형대수에 익숙하지 않으면 [2장](02-linear-algebra-dirac.md)과 [선형대수 노트](../../math/linear-algebra/index.md)를 먼저 읽습니다.
2. 알고리즘이 목적이면 1~4장에서 회로와 보편 게이트의 최소 규칙만 익히고 [6장](06-intro-algorithms.md)으로 넘어갑니다.
3. 각 장은 추상 정리를 적은 뒤 행렬 원소를 직접 곱하거나 작은 크기의 회로($N=15$, $n=4$ 위상 추정, $9$-큐비트 $Y_1$ 오류 정정)를 끝까지 계산합니다.

---

## 다른 섹션과의 관계

- [선형대수](../../math/linear-algebra/index.md): 내적 공간, 에르미트·유니타리 행렬, 특잇값 분해의 기초 정리를 다룹니다.
- [확률론](../../math/probability/index.md): 확률 변수, 기댓값, 마르코프·체비쇼프·체르노프 부등식의 일반 증명을 제공합니다.
- [알고리즘](../algorithms/index.md): 고전 고속 푸리에 변환([30장](../algorithms/30-fft.md)), 정수론과 모듈러 연산([31장](../algorithms/31-number-theory.md)), `P`/`NP`([34장](../algorithms/34-np-completeness.md))를 비교할 수 있습니다.
- [양자역학](../../physics/quantum-mechanics/index.md): 연속 공간의 슈뢰딩거 방정식, 각운동량, 스핀-1/2 물리 계와 [벨 정리(12장)](../../physics/quantum-mechanics/12-afterword.md)를 다룹니다.
