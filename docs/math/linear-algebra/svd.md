---
title: 특이값 분해
tags:
  - math
  - linear-algebra
  - low-rank
---

# 특이값 분해

## 정의

임의의 행렬 $A \in \mathbb{R}^{m \times n}$ 은 다음과 같이 분해할 수 있습니다.

$$
A = U \Sigma V^{\top}
$$

여기서 $U \in \mathbb{R}^{m \times m}$ 와 $V \in \mathbb{R}^{n \times n}$ 은 직교 행렬이고,
$\Sigma \in \mathbb{R}^{m \times n}$ 는 대각 성분에 특이값
$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 을 갖는 행렬입니다.

## 저랭크 근사

에카르트-영 정리에 따르면, 랭크가 $k$ 이하인 행렬 중에서 $A$ 와의 오차를 최소화하는
행렬은 상위 $k$ 개의 특이값만 남긴 다음 식으로 주어집니다.

$$
A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\top}, \qquad
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^{2}}
$$

!!! tip "LoRA와의 연결"

    사전 학습된 가중치의 갱신량 $\Delta W$ 가 실제로는 낮은 랭크를 갖는다는 가정 위에서,
    $\Delta W = BA$ 형태로 두 개의 작은 행렬만 학습하는 기법이 LoRA입니다.
    위 식은 그러한 근사가 어느 정도의 오차를 남기는지를 정량적으로 알려 줍니다.

## 계산 예시

```python
import numpy as np

A = np.random.randn(64, 32)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

k = 8
A_k = (U[:, :k] * s[:k]) @ Vt[:k, :]
print(np.linalg.norm(A - A_k, "fro"), np.sqrt((s[k:] ** 2).sum()))
```
