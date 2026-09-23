---
title: 다층 퍼셉트론
tags:
  - books
  - llm
  - neural-network
---

# 1. 다층 퍼셉트론

## 뉴런 하나가 하는 계산

뉴런 하나는 입력 벡터에 가중치를 곱해서 모두 더한 뒤, 편향을 더하고,
그 결과를 활성화 함수에 통과시킵니다. 입력을 $x \in \mathbb{R}^{d}$,
가중치를 $w \in \mathbb{R}^{d}$, 편향을 $b \in \mathbb{R}$ 라고 하면 다음과 같습니다.

$$
z = w^{\top} x + b, \qquad a = \phi(z)
$$

여기서 $z$ 를 사전 활성화(pre-activation), $a$ 를 활성화(activation)라고 부릅니다.
이 둘을 구분해서 이름을 붙여 두면, 뒤에서 역전파를 계산할 때 어느 값에 대한
미분인지 헷갈리지 않습니다.

## 층 단위로 묶으면 행렬 곱이 됩니다

뉴런을 여러 개 두면 가중치 벡터도 그만큼 늘어납니다. 이때 가중치 벡터들을
열로 쌓아서 행렬 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 을 만들면,
뉴런 전체의 계산을 행렬 곱 한 번으로 처리할 수 있습니다.

$$
z = x^{\top} W + b, \qquad z \in \mathbb{R}^{d_{\text{out}}}
$$

여기서 반복문 대신 행렬 곱을 쓰는 이유는 단순히 코드가 짧아지기 때문이 아닙니다.
행렬 곱은 BLAS 수준에서 고도로 최적화되어 있고 GPU에서 병렬로 실행되기 때문에,
같은 계산이라도 수십 배 이상 빨라집니다.

## 배치를 한 번에 처리합니다

샘플 $N$ 개를 행으로 쌓아서 $X \in \mathbb{R}^{N \times d_{\text{in}}}$ 을 만들면,
층 전체의 계산은 다음 한 줄이 됩니다.

$$
Z = XW + b
$$

$b$ 의 형태는 $(d_{\text{out}},)$ 이고 $XW$ 의 형태는 $(N, d_{\text{out}})$ 이라서
차원이 맞지 않아 보이지만, 브로드캐스팅 덕분에 $b$ 가 모든 행에 자동으로 더해집니다.

## 실행 코드

```python title="mlp.py"
import numpy as np

rng = np.random.default_rng(0)

# --- 뉴런 하나 ---
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -1.0, 0.25])
b = 0.1
print("[뉴런 1개]")
print("  z = w·x + b =", w @ x + b)

# --- 층 하나: 입력 3차원 -> 뉴런 4개 ---
W1 = rng.normal(0, 0.5, size=(3, 4))
b1 = np.zeros(4)
print("\n[층 1개] 입력 3차원 -> 뉴런 4개")
print("  W1.shape =", W1.shape, " b1.shape =", b1.shape)
print("  z1 =", np.round(x @ W1 + b1, 4))

# --- 배치 처리: 샘플 5개를 한 번에 ---
X = rng.normal(0, 1, size=(5, 3))
Z1 = X @ W1 + b1
print("\n[배치 처리] 샘플 5개")
print("  X.shape =", X.shape, "@ W1.shape =", W1.shape, "-> Z1.shape =", Z1.shape)
print("  b1은 (4,)이지만 브로드캐스팅으로 5개 행에 모두 더해집니다.")

# --- 2층 MLP 순전파 ---
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

d_in, d_hidden, d_out = 3, 4, 2
W1 = rng.normal(0, 0.5, size=(d_in, d_hidden)); b1 = np.zeros(d_hidden)
W2 = rng.normal(0, 0.5, size=(d_hidden, d_out)); b2 = np.zeros(d_out)

H = relu(X @ W1 + b1)
P = softmax(H @ W2 + b2)

print("\n[2층 MLP 순전파]")
print("  X (5,3) -> H (5,4) -> P (5,2)")
print("  H =\n", np.round(H, 4))
print("  P =\n", np.round(P, 4))
print("  각 행의 합 =", np.round(P.sum(axis=1), 10))

n_params = W1.size + b1.size + W2.size + b2.size
print("\n[파라미터 수]")
print(f"  W1 {W1.size} + b1 {b1.size} + W2 {W2.size} + b2 {b2.size} = {n_params}")
```

## 실행 결과

```text
[뉴런 1개]
  z = w·x + b = -0.65

[층 1개] 입력 3차원 -> 뉴런 4개
  W1.shape = (3, 4)  b1.shape = (4,)
  z1 = [-1.5284 -1.6026  0.6893  1.0615]

[배치 처리] 샘플 5개
  X.shape = (5, 3) @ W1.shape = (3, 4) -> Z1.shape = (5, 4)
  b1은 (4,)이지만 브로드캐스팅으로 5개 행에 모두 더해집니다.

[2층 MLP 순전파]
  X (5,3) -> H (5,4) -> P (5,2)
  H =
 [[0.873  0.9568 0.     0.6618]
 [0.3382 0.3143 0.     0.1873]
 [0.     0.     0.3356 0.    ]
 [0.     0.     0.     0.    ]
 [0.     0.0288 0.1731 0.    ]]
  P =
 [[0.6672 0.3328]
 [0.5808 0.4192]
 [0.5217 0.4783]
 [0.5    0.5   ]
 [0.5118 0.4882]]
  각 행의 합 = [1. 1. 1. 1. 1.]

[파라미터 수]
  W1 12 + b1 4 + W2 8 + b2 2 = 26
```

## 결과에서 짚어 볼 점

**`H` 의 4번째 행이 전부 0입니다.** 이 샘플은 은닉층의 모든 뉴런에서
사전 활성화가 음수였기 때문에, ReLU를 통과하면서 신호가 완전히 사라졌습니다.
그 결과 `P` 의 4번째 행은 정확히 `[0.5, 0.5]` 가 되었습니다.
입력이 무엇이든 은닉층이 0을 내보내면 출력층은 편향만 보게 되므로,
모델이 아무 정보도 얻지 못한 상태가 됩니다.

이 현상이 층 전체에서 지속되면 죽은 ReLU(dying ReLU) 문제가 됩니다.
다음 장의 Leaky ReLU는 바로 이 문제를 완화하려는 시도입니다.

**`P` 의 각 행의 합이 정확히 1입니다.** softmax가 출력을 확률 분포로 만들어 주기 때문입니다.
합이 1이라는 제약은 뒤에서 야코비안을 계산할 때 각 행의 합이 0이 되는 성질로 이어집니다.

## 파라미터 수를 세는 습관

층의 파라미터 수는 $d_{\text{in}} \times d_{\text{out}} + d_{\text{out}}$ 입니다.
위 예시는 $3 \times 4 + 4 + 4 \times 2 + 2 = 26$ 개였습니다.

이 계산을 몸에 익혀 두면 모델 크기를 어림잡을 때 유용합니다.
예를 들어 $d_{\text{model}} = 4096$, $d_{\text{ff}} = 11008$ 인 트랜스포머 블록의
피드포워드 층 하나는 가중치 행렬만으로도 약 4500만 개의 파라미터를 갖습니다.
편향은 전체의 0.1퍼센트도 되지 않기 때문에, 최근 모델들이 편향을 아예 제거하는
선택을 해도 크기에는 거의 영향이 없습니다.
