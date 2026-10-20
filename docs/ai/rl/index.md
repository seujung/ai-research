---
title: 강화 학습
tags:
  - reinforcement-learning
  - sutton-barto
---

# 강화 학습

Sutton과 Barto의 *Reinforcement Learning: An Introduction* 2판을 따라 강화 학습의 핵심 개념을 정리합니다.
다중 선택 밴딧과 유한 마르코프 결정 과정에서 시작해 동적 계획법, 몬테카를로 방법, 시간차 학습을 거칩니다.
이어서 함수 근사와 정책 그래디언트로 넘어가고, 심리학·신경과학과의 연결과 응용 사례로 끝납니다.

각 장은 원문을 번역하지 않습니다. 개념마다 정의(수식), 직관, 작은 수치 예시를 붙이고,
원문의 대표 예제를 직접 구현해 그림으로 확인합니다.

!!! info "원문"
    - 책: Richard S. Sutton, Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
    - 기준 원고: [2판 초안(2015)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
    - 보충 원고: [2판 최종본(2018)](http://incompleteideas.net/book/RLbook2020.pdf), [책 페이지](http://incompleteideas.net/book/the-book-2nd.html)
    - 초안은 off-policy 근사, 정책 그래디언트, 심리학과 신경과학 장이 비어 있거나 짧습니다. 이 부분은 최종본으로 보충했습니다.
    - 기호는 최종본 표기($S_t, A_t, R_{t+1}, G_t, v_\pi, q_\pi$)를 씁니다. 이 노트의 본문과 그림은 원문을 참고해 새로 작성했습니다.

## 목차

| 장 | 원문(초안 / 최종본) | 다루는 내용 |
| --- | --- | --- |
| [1. 강화 학습 문제](01-rl-problem.md) | 1장 / 1장 | 정의, 탐험과 활용, 구성 요소, 틱택토, 역사 |
| [2. 다중 선택 밴딧](02-bandits.md) | 2장 / 2장 | ε-탐욕, 증분 평균, 낙관적 초기값, UCB, 그래디언트 밴딧 |
| [3. 유한 마르코프 결정 과정](03-finite-mdp.md) | 3장 / 3장 | 이득과 할인, 마르코프 성질, 벨만 방정식, 최적 가치 함수 |
| [4. 동적 계획법](04-dynamic-programming.md) | 4장 / 4장 | 정책 평가, 정책 개선 정리, 정책 반복, 가치 반복, GPI |
| [5. 몬테카를로 방법](05-monte-carlo.md) | 5장 / 5장 | MC 예측과 제어, 탐험적 시작, 중요도 샘플링 |
| [6. 시간차 학습](06-td-learning.md) | 6장 / 6장 | TD(0), Sarsa, Q-learning, 기대 Sarsa, 이중 Q-learning |
| [7. n단계 TD와 적격 흔적](07-eligibility-traces.md) | 7장 / 7장, 12장 | n단계 이득, λ-이득, 전방·후방 관점, Sarsa(λ) |
| [8. 계획과 학습](08-planning-learning.md) | 8장 / 8장 | 모델, Dyna-Q, 우선순위 쓸기, 궤적 샘플링, MCTS |
| [9. 함수 근사](09-function-approximation.md) | 9장 / 9장, 10장 | 반경사 방법, 선형 TD 고정점, 타일 부호화, 산악 자동차 |
| [10. 근사를 쓰는 off-policy 학습](10-off-policy-approximation.md) | - / 11장 | 치명적 삼중주, Baird 반례, 투영 벨만 오차, 그래디언트 TD |
| [11. 정책 근사](11-policy-gradient.md) | 11장 / 13장, 10장 | 정책 그래디언트 정리, REINFORCE, 액터-크리틱, 평균 보상 |
| [12. 심리학과 신경과학](12-psychology-neuroscience.md) | - / 14장, 15장 | Rescorla-Wagner, TD 모형, 도파민과 보상 예측 오차 |
| [13. 응용 사례와 전망](13-applications-frontiers.md) | 14장, 15장 / 16장, 17장 | TD-Gammon, DQN, AlphaGo, 일반 가치 함수, 옵션 |
| [14. 선호 학습과 RLHF](14-preference-rlhf.md) | 책 밖 (논문) | 보상 모델, KL 정규화, RLHF-PPO, DPO, IPO, KTO, SimPO |
| [15. 검증 가능한 보상의 강화 학습](15-rlvr-grpo.md) | 책 밖 (논문) | RLOO, GRPO, Dr. GRPO, DAPO, GSPO |

## 읽는 방법

1~6장은 표 형식(tabular) 방법의 기본입니다. 순서대로 읽는 편이 좋습니다.
7장과 8장은 6장을 두 방향으로 넓힙니다. 7장은 MC와 TD 사이를 잇고, 8장은 학습과 계획을 잇습니다.
9~11장은 상태가 너무 많아 표를 쓸 수 없을 때의 방법입니다. 9장을 먼저 읽고 10장과 11장으로 갑니다.
12장과 13장은 앞 장의 알고리즘이 동물 학습과 실제 문제에서 어떻게 나타나는지 다룹니다.
14장과 15장은 책에 없는 내용입니다. 대형 언어 모델(LLM)의 후속 학습에 쓰는 최근 방법을 원 논문을 기준으로 정리합니다. 11장의 정책 그래디언트와 PPO를 먼저 읽습니다.

9장 이후는 [신경망 기초](../d2l/index.md)의 경사 하강법과 [확률론](../../math/probability/index.md)의 기댓값을 전제로 합니다.
본문의 수치 예시와 실험 곡선은 모두 NumPy로 직접 계산해 확인했습니다.
