---
title: 확률론
tags:
  - math
  - probability
---

# 확률론

## 다룰 주제

- 확률 분포와 기댓값
- 최대 우도 추정
- 쿨백-라이블러 발산과 교차 엔트로피

## 교차 엔트로피가 언어 모델의 손실 함수인 이유

정답 분포 $p$ 와 모델의 예측 분포 $q$ 사이의 쿨백-라이블러 발산은 다음과 같습니다.

$$
D_{\mathrm{KL}}(p \parallel q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
= \underbrace{-\sum_{x} p(x) \log q(x)}_{H(p,\,q)} - \underbrace{\left(-\sum_{x} p(x) \log p(x)\right)}_{H(p)}
$$

정답 분포 $p$ 는 학습 과정에서 고정되어 있으므로 $H(p)$ 도 상수가 됩니다.
따라서 쿨백-라이블러 발산을 최소화하는 문제는 교차 엔트로피 $H(p, q)$ 를
최소화하는 문제와 완전히 같아집니다.
