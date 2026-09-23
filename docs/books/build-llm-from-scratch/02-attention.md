---
title: 어텐션
tags:
  - books
  - llm
  - attention
---

# 2. 어텐션

## 스케일드 닷 프로덕트 어텐션

쿼리 $Q \in \mathbb{R}^{n \times d_k}$, 키 $K \in \mathbb{R}^{m \times d_k}$,
값 $V \in \mathbb{R}^{m \times d_v}$ 가 주어졌을 때 어텐션은 다음과 같이 정의됩니다.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

### $\sqrt{d_k}$ 로 나누는 이유

$q$ 와 $k$ 의 각 성분이 평균 0, 분산 1인 독립 확률 변수라고 가정하면,
내적 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 의 분산은 $d_k$ 가 됩니다.
따라서 $d_k$ 가 커질수록 내적 값의 범위가 넓어지고, 소프트맥스의 입력이
극단으로 몰리면서 기울기가 0에 가까워집니다.
표준편차인 $\sqrt{d_k}$ 로 나누면 분산이 1로 유지되므로 이 문제를 완화할 수 있습니다.

## 인과적 마스킹

디코더에서는 현재 위치보다 뒤에 있는 토큰을 참조하면 안 되기 때문에,
소프트맥스를 적용하기 전에 상삼각 영역을 $-\infty$ 로 채웁니다.

```python
import torch

scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```
