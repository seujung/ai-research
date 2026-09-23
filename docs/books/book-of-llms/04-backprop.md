---
title: 역전파
tags:
  - books
  - llm
  - neural-network
  - backpropagation
---

# 4. 역전파

## 연쇄 법칙을 계산 그래프에 적용합니다

역전파는 새로운 수학이 아니라 연쇄 법칙을 효율적인 순서로 적용하는 기법입니다.

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

핵심은 **계산 순서**에 있습니다. 출력에서 시작해서 입력 방향으로 거슬러 올라가면,
각 노드는 자기 뒤에서 흘러온 기울기 하나만 받아서 자기 몫의 미분을 곱하면 됩니다.
반대로 입력에서 출력 방향으로 계산하면 파라미터 개수만큼 전체 그래프를 훑어야 합니다.
파라미터가 수십억 개인 모델에서 이 차이는 계산 가능 여부를 가릅니다.

## 2층 MLP의 역전파 유도

순전파는 다음과 같습니다.

$$
Z_1 = XW_1 + b_1, \quad H = \mathrm{ReLU}(Z_1), \quad
Z_2 = HW_2 + b_2, \quad P = \mathrm{softmax}(Z_2)
$$

손실은 평균 교차 엔트로피를 씁니다.

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \log P_{n, y_n}
$$

### 출력층에서 시작합니다

Softmax와 교차 엔트로피를 따로 미분하면 야코비안이 등장해서 복잡해집니다.
그런데 둘을 하나로 묶으면 놀랍도록 단순한 식이 나옵니다.

$$
\frac{\partial L}{\partial Z_2} = \frac{1}{N}(P - Y_{\text{onehot}})
$$

예측 확률에서 정답을 빼기만 하면 됩니다. 앞 장에서 구한 softmax 야코비안
$\mathrm{diag}(p) - pp^{\top}$ 와 로그의 미분 $-1/p_y$ 가 곱해지면서
대부분의 항이 상쇄되기 때문입니다. 이 단순화 덕분에 실제 구현에서는
두 연산을 항상 하나의 함수로 합쳐서 제공합니다.

### 나머지는 기계적으로 따라갑니다

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= H^{\top} \frac{\partial L}{\partial Z_2},
&\frac{\partial L}{\partial b_2} &= \sum_{n} \frac{\partial L}{\partial Z_2}\bigg|_{n} \\[4pt]
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial Z_2} W_2^{\top},
&\frac{\partial L}{\partial Z_1} &= \frac{\partial L}{\partial H} \odot \mathbb{1}[Z_1 > 0] \\[4pt]
\frac{\partial L}{\partial W_1} &= X^{\top} \frac{\partial L}{\partial Z_1},
&\frac{\partial L}{\partial b_1} &= \sum_{n} \frac{\partial L}{\partial Z_1}\bigg|_{n}
\end{aligned}
$$

!!! tip "형태를 보고 식을 복원하는 요령"

    미분식을 외우기보다 형태를 맞춰 보는 편이 빠릅니다.
    $\partial L / \partial W_2$ 는 $W_2$ 와 같은 $(d_h, d_{\text{out}})$ 이어야 하는데,
    $H$ 는 $(N, d_h)$ 이고 $\partial L / \partial Z_2$ 는 $(N, d_{\text{out}})$ 입니다.
    이 둘로 $(d_h, d_{\text{out}})$ 을 만드는 조합은 $H^{\top} \cdot \partial L/\partial Z_2$ 하나뿐입니다.

    편향의 기울기에서 배치 차원으로 합을 취하는 이유도 같습니다.
    $b$ 는 $(d_{\text{out}},)$ 인데 기울기는 $(N, d_{\text{out}})$ 으로 들어오므로,
    형태를 맞추려면 배치 축을 없애야 합니다. 순전파에서 브로드캐스팅으로
    복제된 값은 역전파에서 합으로 되돌아옵니다.

## 그래디언트 검증

구현이 맞는지는 수치 미분과 비교해서 확인합니다.
상대 오차가 $10^{-7}$ 이하이면 신뢰할 만합니다.

```python title="gradcheck.py"
import numpy as np
np.set_printoptions(precision=4, suppress=True)

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])

def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def init(d_in=2, d_h=4, d_out=2, seed=7, bias_scale=0.0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 1.0, (d_in, d_h)), "b1": r.normal(0, bias_scale, d_h),
            "W2": r.normal(0, 1.0, (d_h, d_out)), "b2": r.normal(0, bias_scale, d_out)}

def forward(p, X):
    Z1 = X @ p["W1"] + p["b1"]; H = relu(Z1)
    Z2 = H @ p["W2"] + p["b2"]; P = softmax(Z2)
    return {"Z1": Z1, "H": H, "Z2": Z2, "P": P}

def loss_fn(p, X, y):
    P = forward(p, X)["P"]
    return -np.log(P[np.arange(len(y)), y] + 1e-12).mean()

def backward(p, c, X, y):
    N = len(y)
    dZ2 = c["P"].copy(); dZ2[np.arange(N), y] -= 1; dZ2 /= N
    dW2 = c["H"].T @ dZ2; db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * relu_grad(c["Z1"])
    return {"W1": X.T @ dZ1, "b1": dZ1.sum(axis=0), "W2": dW2, "b2": db2}

def grad_check(p, h=1e-6):
    g = backward(p, forward(p, X), X, y)
    out = {}
    for name in ["W1", "b1", "W2", "b2"]:
        num = np.zeros_like(p[name])
        it = np.nditer(p[name], flags=["multi_index"])
        while not it.finished:
            i = it.multi_index; o = p[name][i]
            p[name][i] = o + h; lp = loss_fn(p, X, y)
            p[name][i] = o - h; lm = loss_fn(p, X, y)
            p[name][i] = o
            num[i] = (lp - lm) / (2 * h); it.iternext()
        out[name] = np.abs(g[name] - num).max() / max(np.abs(num).max(), 1e-12)
    return out

print("[검증 1] 편향을 0으로 초기화한 경우")
p = init(bias_scale=0.0)
for k, v in grad_check(p).items():
    print(f"  {k:<3} 상대 오차 = {v:.3e}  {'통과' if v < 1e-6 else '실패'}")

Z1 = X @ p["W1"] + p["b1"]
print("\n  원인을 찾기 위해 Z1을 확인합니다:")
print("  Z1 =\n", Z1)
print(f"  정확히 0인 원소가 {int((Z1 == 0).sum())}개 있습니다. 모두 입력이 [0, 0]인 첫 번째 행입니다.")

print("\n[검증 2] 편향을 작은 난수로 초기화해서 0을 피한 경우")
p = init(bias_scale=0.1)
Z1 = X @ p["W1"] + p["b1"]
print(f"  Z1에서 정확히 0인 원소: {int((Z1 == 0).sum())}개")
for k, v in grad_check(p).items():
    print(f"  {k:<3} 상대 오차 = {v:.3e}  {'통과' if v < 1e-6 else '실패'}")
```

```text
[검증 1] 편향을 0으로 초기화한 경우
  W1  상대 오차 = 8.277e-10  통과
  b1  상대 오차 = 8.084e-01  실패
  W2  상대 오차 = 1.531e-09  통과
  b2  상대 오차 = 6.397e-10  통과

  원인을 찾기 위해 Z1을 확인합니다:
  Z1 =
 [[ 0.      0.      0.      0.    ]
 [-0.4547 -0.9916  0.0601  1.3402]
 [ 0.0012  0.2987 -0.2741 -0.8906]
 [-0.4534 -0.6929 -0.214   0.4496]]
  정확히 0인 원소가 4개 있습니다. 모두 입력이 [0, 0]인 첫 번째 행입니다.

[검증 2] 편향을 작은 난수로 초기화해서 0을 피한 경우
  Z1에서 정확히 0인 원소: 0개
  W1  상대 오차 = 3.707e-10  통과
  b1  상대 오차 = 3.522e-10  통과
  W2  상대 오차 = 1.181e-09  통과
  b2  상대 오차 = 3.403e-10  통과
```

!!! danger "검증 실패가 항상 구현 오류를 뜻하지는 않습니다"

    첫 번째 검증에서 `b1` 만 상대 오차가 0.8로 나왔습니다. 그런데 역전파 구현에는
    아무 문제가 없었습니다. 원인은 **ReLU가 정확히 0인 지점에서 미분이 정의되지 않는다**는
    점에 있었습니다.

    입력 `[0, 0]` 인 샘플은 $XW_1$ 이 0이고 편향도 0이므로 $Z_1$ 이 정확히 0이 됩니다.
    이 상태에서 편향을 $+h$ 로 흔들면 뉴런이 살아나고 $-h$ 로 흔들면 죽기 때문에,
    수치 미분이 한쪽으로 치우친 값을 내놓습니다. 반면 해석적 미분은
    `(z > 0)` 이라는 판정에 따라 0을 반환합니다. 둘이 어긋나는 것이 당연합니다.

    편향을 작은 난수로 초기화해서 꺾인 지점을 피하자 네 파라미터가 모두 통과했습니다.
    ReLU 계열을 쓸 때 그래디언트 검증이 간헐적으로 실패한다면, 구현을 의심하기 전에
    먼저 사전 활성화에 0이 섞여 있는지 확인하는 편이 좋습니다.

## 학습이 실제로 되는지 확인합니다

XOR은 선형 분리가 불가능한 가장 작은 문제입니다.
은닉층이 없으면 절대 풀 수 없으므로, 층을 쌓는 효과를 확인하기에 적합합니다.

```python title="train.py"
print("[학습] 은닉층 4개, 학습률 0.5")
p = init()
lr = 0.5
print(f"  {'step':>5}  {'loss':>8}  {'정확도':>6}")
for step in range(2001):
    cache = forward(p, X)
    g = backward(p, cache, X, y)
    for k in p:
        p[k] -= lr * g[k]
    if step % 400 == 0:
        L = loss_fn(p, X, y)
        acc = (forward(p, X)["P"].argmax(axis=1) == y).mean()
        print(f"  {step:>5}  {L:>8.5f}  {acc:>6.2f}")

print("\n[최종 예측]")
P = forward(p, X)["P"]
for xi, yi, pi in zip(X, y, P):
    print(f"  입력 {xi} | 정답 {yi} | 예측 {pi.argmax()} | 확률 {pi}")

# --- 대조군: 은닉층이 없으면 XOR을 풀 수 없습니다 ---
print("\n[대조군] 은닉층 없이 선형 분류기만 사용한 경우")
rng = np.random.default_rng(7)
W = rng.normal(0, 1.0, (2, 2)); b = np.zeros(2)
for _ in range(2001):
    P = softmax(X @ W + b)
    dZ = P.copy(); dZ[np.arange(4), y] -= 1; dZ /= 4
    W -= 0.5 * (X.T @ dZ); b -= 0.5 * dZ.sum(axis=0)
P = softmax(X @ W + b)
L = -np.log(P[np.arange(4), y] + 1e-12).mean()
print(f"  loss = {L:.5f}, 정확도 = {(P.argmax(axis=1) == y).mean():.2f}")
print("  네 샘플 모두 확률이 0.5 근처에 머무릅니다:\n ", np.round(P, 4))
```

```text
[학습] 은닉층 4개, 학습률 0.5
   step      loss     정확도
      0   0.63356    0.50
    400   0.00521    1.00
    800   0.00232    1.00
   1200   0.00147    1.00
   1600   0.00107    1.00
   2000   0.00084    1.00

[최종 예측]
  입력 [0. 0.] | 정답 0 | 예측 0 | 확률 [0.9986 0.0014]
  입력 [0. 1.] | 정답 1 | 예측 1 | 확률 [0.0003 0.9997]
  입력 [1. 0.] | 정답 1 | 예측 1 | 확률 [0.0003 0.9997]
  입력 [1. 1.] | 정답 0 | 예측 0 | 확률 [0.9986 0.0014]

[대조군] 은닉층 없이 선형 분류기만 사용한 경우
  loss = 0.69315, 정확도 = 0.50
  네 샘플 모두 확률이 0.5 근처에 머무릅니다:
  [[0.5 0.5]
 [0.5 0.5]
 [0.5 0.5]
 [0.5 0.5]]
```

## 결과에서 짚어 볼 점

**대조군의 손실이 정확히 0.69315입니다.** 이 값은 $\ln 2 = 0.693147\ldots$ 입니다.
두 클래스에 대해 아무것도 모르는 모델이 낼 수 있는 손실의 값이 바로 $\ln 2$ 이므로,
선형 분류기는 2000번을 학습하고도 **동전 던지기와 완전히 같은 상태에 머물렀습니다.**

여기서 중요한 점은 모델이 발산하거나 오류를 내지 않았다는 사실입니다.
손실은 매끄럽게 수렴했고 학습은 정상적으로 끝났습니다. 다만 도달한 지점이
무의미한 해였을 뿐입니다. 손실 곡선이 안정적으로 내려간다는 사실만으로는
모델이 문제를 풀고 있다고 말할 수 없으며, 반드시 무작위 추측의 기준선과
비교해야 하는 이유입니다.

$k$ 개 클래스를 균등하게 예측할 때의 교차 엔트로피는 $\ln k$ 입니다.
분류 문제를 학습시킬 때 이 값을 먼저 계산해 두면, 학습이 시작된 직후의 손실이
정상 범위인지 즉시 판단할 수 있습니다.

**은닉층을 넣은 모델은 400스텝 만에 정확도 1.0에 도달했습니다.**
은닉층의 ReLU가 입력 공간을 접어서, 원래는 선형 분리가 불가능하던 네 점을
분리 가능한 배치로 바꾸어 놓았기 때문입니다. 층을 쌓는다는 것은
표현을 반복적으로 변형해서 마지막 선형 층이 풀 수 있는 형태로 만드는 작업입니다.
