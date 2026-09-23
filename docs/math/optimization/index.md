---
title: 최적화
tags:
  - math
  - optimization
---

# 최적화

## 다룰 주제

- 경사 하강법과 확률적 경사 하강법
- 모멘텀과 적응적 학습률
- 학습률 스케줄링

## Adam의 갱신 규칙

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^{2} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

$m_0$ 와 $v_0$ 를 0으로 초기화하면 학습 초반에 두 값이 0 쪽으로 치우치게 됩니다.
$1 - \beta^{t}$ 로 나누는 편향 보정 항은 바로 이 치우침을 제거하는 역할을 합니다.
