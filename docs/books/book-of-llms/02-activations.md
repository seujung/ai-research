---
title: 활성화 함수
tags:
  - books
  - llm
  - neural-network
  - activation
---

# 2. 활성화 함수

## 왜 필요한가

선형 변환을 아무리 여러 번 쌓아도 결국 하나의 선형 변환으로 합쳐집니다.

$$
W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = \tilde{W} x + \tilde{b}
$$

따라서 활성화 함수가 없으면 층을 100개 쌓아도 표현력은 1층짜리 선형 모델과 같습니다.
비선형 함수를 중간에 끼워 넣어야 비로소 층을 쌓는 의미가 생깁니다.

## 주요 함수들

| 함수 | 정의 | 치역 | 특징 |
| --- | --- | --- | --- |
| 시그모이드 | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $(0, 1)$ | 양 끝에서 포화되어 기울기가 사라집니다. |
| 하이퍼볼릭 탄젠트 | $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $(-1, 1)$ | 0을 중심으로 대칭이지만 역시 포화됩니다. |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | 양수 구간에서 기울기가 1로 유지됩니다. |
| Leaky ReLU | $\max(\alpha z, z)$ | $(-\infty, \infty)$ | 음수 구간에도 작은 기울기를 남겨 둡니다. |
| Swish | $z \cdot \sigma(\beta z)$ | $\approx (-0.28, \infty)$ | 매끄럽고 단조 증가가 아닙니다. |

시그모이드와 탄젠트의 도함수는 다음과 같이 자기 자신으로 표현됩니다.

$$
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr), \qquad \tanh'(z) = 1 - \tanh^{2}(z)
$$

$\sigma(z)$ 는 $(0, 1)$ 안에 있으므로 $\sigma'(z)$ 의 최댓값은 $z = 0$ 에서 $0.25$ 입니다.
층을 $L$ 개 쌓으면 역전파 과정에서 이 값이 $L$ 번 곱해지므로, 기울기는 최소한
$0.25^{L}$ 배로 줄어듭니다. 층이 10개만 되어도 $10^{-6}$ 수준이 됩니다.
이것이 기울기 소실(vanishing gradient) 문제의 핵심입니다.

## 실행 코드

```python title="act.py"
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def sigmoid(z):  return 1 / (1 + np.exp(-z))
def tanh(z):     return np.tanh(z)
def relu(z):     return np.maximum(0, z)
def leaky_relu(z, a=0.01): return np.where(z > 0, z, a * z)
def swish(z, beta=1.0):    return z * sigmoid(beta * z)

z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("입력 z          =", z)
print("sigmoid(z)      =", sigmoid(z))
print("tanh(z)         =", tanh(z))
print("relu(z)         =", relu(z))
print("leaky_relu(z)   =", leaky_relu(z))
print("swish(z)        =", swish(z))

print("\n[포화 구간에서 기울기가 사라지는 현상]")
print(" z      sigmoid'(z)      tanh'(z)")
for v in [0.0, 2.0, 5.0, 10.0, 20.0]:
    s = sigmoid(v)
    print(f"{v:5.1f}   {s*(1-s):12.3e}   {1-np.tanh(v)**2:12.3e}")
```

## 실행 결과

```text
입력 z          = [-3. -1.  0.  1.  3.]
sigmoid(z)      = [0.0474 0.2689 0.5    0.7311 0.9526]
tanh(z)         = [-0.9951 -0.7616  0.      0.7616  0.9951]
relu(z)         = [0. 0. 0. 1. 3.]
leaky_relu(z)   = [-0.03 -0.01  0.    1.    3.  ]
swish(z)        = [-0.1423 -0.2689  0.      0.7311  2.8577]

[포화 구간에서 기울기가 사라지는 현상]
 z      sigmoid'(z)      tanh'(z)
  0.0      2.500e-01      1.000e+00
  2.0      1.050e-01      7.065e-02
  5.0      6.648e-03      1.816e-04
 10.0      4.540e-05      8.245e-09
 20.0      2.061e-09      0.000e+00
```

### 결과에서 짚어 볼 점

$z = 20$ 에서 $\tanh'(z)$ 가 **정확히 0으로 출력되었습니다.** 수학적으로는 0이 아니라
$10^{-17}$ 정도의 작은 양수이지만, 배정밀도 부동소수점의 표현 한계를 넘어서면서
완전히 0이 되어 버렸습니다. 이 상태에 빠진 뉴런은 역전파로 어떤 신호도 받지 못하므로,
학습이 영원히 멈춥니다.

Swish가 음수 구간에서 $-0.2689$ 처럼 음수 값을 내는 점도 눈여겨볼 만합니다.
$z = -3$ 일 때 $-0.1423$, $z = -1$ 일 때 $-0.2689$ 이므로, 입력이 작아지는데
출력은 오히려 커졌습니다. 즉 단조 증가 함수가 아닙니다.
이 성질이 최적화에 도움이 된다는 것이 Swish를 제안한 연구의 실험적 발견이었습니다.

## Softmax와 수치 안정성

Softmax는 벡터를 확률 분포로 바꿉니다.

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

구현할 때 반드시 최댓값을 빼야 합니다. 분자와 분모에 같은 상수 $e^{-c}$ 를 곱하는
것이므로 결과는 수학적으로 동일하지만, 지수 함수가 넘치는 것을 막아 줍니다.

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i - c}}{\sum_{j} e^{z_j - c}}, \qquad c = \max_j z_j
$$

```python title="softmax.py"
import numpy as np, warnings
np.set_printoptions(precision=6, suppress=True)

def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.array([1.0, 2.0, 3.0])
print("[일반적인 입력]")
print("  naive  =", softmax_naive(z))
print("  stable =", softmax_stable(z))

z_big = np.array([1000.0, 1001.0, 1002.0])
print("\n[큰 값이 들어온 경우]")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("  naive  =", softmax_naive(z_big), " <- exp(1000)이 inf로 넘쳐서 nan이 됩니다")
print("  stable =", softmax_stable(z_big), " <- 최댓값을 빼면 정상 동작합니다")
print("\n  exp(1000) =", np.exp(np.float64(1000.0)))
print("  두 입력의 차이가 같으므로 결과도 같아야 합니다:",
      np.allclose(softmax_stable(z), softmax_stable(z_big)))

print("\n[온도(temperature)에 따른 분포 변화]")
logits = np.array([2.0, 1.0, 0.5, 0.1])
for T in [0.5, 1.0, 2.0, 10.0]:
    print(f"  T={T:<5} -> {softmax_stable(logits / T)}")
```

```text
[일반적인 입력]
  naive  = [0.090031 0.244728 0.665241]
  stable = [0.090031 0.244728 0.665241]

[큰 값이 들어온 경우]
  naive  = [nan nan nan]  <- exp(1000)이 inf로 넘쳐서 nan이 됩니다
  stable = [0.090031 0.244728 0.665241]  <- 최댓값을 빼면 정상 동작합니다

  exp(1000) = inf
  두 입력의 차이가 같으므로 결과도 같아야 합니다: True

[온도(temperature)에 따른 분포 변화]
  T=0.5   -> [0.828162 0.11208  0.041232 0.018527]
  T=1.0   -> [0.574522 0.211355 0.128193 0.08593 ]
  T=2.0   -> [0.405575 0.245993 0.19158  0.156852]
  T=10.0  -> [0.278357 0.251868 0.239584 0.23019 ]
```

!!! danger "나이브 구현은 조용히 망가집니다"

    `[1000, 1001, 1002]` 와 `[1, 2, 3]` 은 서로의 차이가 동일하므로 softmax 결과도
    같아야 합니다. 그런데 나이브 구현은 `nan` 을 냅니다. 한 번 `nan` 이 생기면
    이후의 모든 연산으로 전파되어 손실이 `nan` 이 되고 학습이 중단됩니다.
    오류 메시지 없이 진행되기 때문에 원인을 찾기가 특히 번거롭습니다.

온도 $T$ 는 로짓을 나누는 값입니다. $T$ 가 작아지면 가장 큰 값에 확률이 몰리고,
$T$ 가 커지면 균등 분포에 가까워집니다. 위 결과에서 $T = 0.5$ 일 때 최고 확률이
0.828이었다가 $T = 10$ 에서 0.278까지 내려가는 것을 확인할 수 있습니다.
LLM에서 생성의 다양성을 조절하는 파라미터가 바로 이것입니다.

## GLU와 SwiGLU

게이트 선형 유닛(GLU)은 입력을 두 갈래의 선형 변환으로 나눈 뒤, 한쪽을 내용으로 쓰고
다른 한쪽을 게이트로 씁니다. 게이트는 내용의 각 차원을 얼마나 통과시킬지 결정합니다.

$$
\mathrm{GLU}(x) = (xW) \odot \sigma(xV)
$$

SwiGLU는 이 게이트의 활성화 함수를 시그모이드 대신 Swish로 바꾼 것입니다.

$$
\mathrm{SwiGLU}(x) = (xW) \odot \mathrm{Swish}(xV)
$$

```python title="glu.py"
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def swish(z):   return z * sigmoid(z)

d_model, d_ff = 8, 16

# 기존 FFN: ReLU를 쓰고 가중치 행렬이 2개입니다.
W_in  = rng.normal(0, 0.3, (d_model, d_ff))
W_out = rng.normal(0, 0.3, (d_ff, d_model))

# GLU 계열: 게이트용 행렬이 하나 더 필요합니다.
W_gate = rng.normal(0, 0.3, (d_model, d_ff))

x = rng.normal(0, 1, (2, d_model))

ffn_relu   = np.maximum(0, x @ W_in) @ W_out
content    = x @ W_in       # 내용(content) 경로
gate       = x @ W_gate     # 게이트(gate) 경로
ffn_glu    = (content * sigmoid(gate)) @ W_out
ffn_swiglu = (content * swish(gate))   @ W_out

print("[게이트가 각 차원을 얼마나 통과시키는가]")
print("  sigmoid(gate)[0] =", sigmoid(gate)[0])
print("  -> 0에 가까운 차원은 거의 차단되고, 1에 가까운 차원은 그대로 통과합니다.")

print("\n[출력 비교] (샘플 0)")
print("  ReLU FFN =", ffn_relu[0])
print("  GLU      =", ffn_glu[0])
print("  SwiGLU   =", ffn_swiglu[0])

print("\n[파라미터 수 비교] d_model=8")
for name, n_mat, dff in [("ReLU FFN", 2, d_ff), ("SwiGLU (같은 d_ff)", 3, d_ff),
                         ("SwiGLU (d_ff를 2/3로)", 3, int(d_ff * 2 / 3))]:
    total = n_mat * d_model * dff
    print(f"  {name:<24} 행렬 {n_mat}개, d_ff={dff:<3} -> {total:>4} 개")
```

```text
[게이트가 각 차원을 얼마나 통과시키는가]
  sigmoid(gate)[0] = [0.6258 0.2323 0.1781 0.6124 0.8257 0.5212 0.5511 0.5636 0.3302 0.6183
 0.3345 0.5617 0.7646 0.1285 0.7087 0.4247]
  -> 0에 가까운 차원은 거의 차단되고, 1에 가까운 차원은 그대로 통과합니다.

[출력 비교] (샘플 0)
  ReLU FFN = [ 0.3007  0.2698  1.5239  1.0935  0.4992 -1.1945 -1.4077 -0.2504]
  GLU      = [-0.1549 -0.2942  0.5043 -0.0969  0.5399 -0.4885  0.0257 -0.1292]
  SwiGLU   = [-0.9152 -0.6902  0.5742 -1.2061  0.5656 -0.2651  1.0518  0.0575]

[파라미터 수 비교] d_model=8
  ReLU FFN                 행렬 2개, d_ff=16  ->  256 개
  SwiGLU (같은 d_ff)         행렬 3개, d_ff=16  ->  384 개
  SwiGLU (d_ff를 2/3로)      행렬 3개, d_ff=10  ->  240 개
```

### 결과에서 짚어 볼 점

게이트 출력을 보면 14번째 차원이 0.1285로 거의 차단되고, 5번째 차원은 0.8257로
대부분 통과합니다. ReLU가 "음수면 전부 버린다"는 이분법을 쓰는 데 비해,
게이트는 **차원마다 0과 1 사이의 연속적인 비율로 통과량을 조절합니다.**
이 표현력의 차이가 GLU 계열이 더 나은 성능을 내는 이유로 설명됩니다.

파라미터 수 비교도 중요합니다. 게이트 행렬이 하나 늘어나므로 같은 $d_{\text{ff}}$ 에서는
파라미터가 1.5배가 됩니다. 그래서 실제 구현에서는 $d_{\text{ff}}$ 를 $2/3$ 로 줄여서
전체 파라미터 수를 비슷하게 맞춥니다. Llama 계열 모델이 $d_{\text{ff}}$ 를
$\frac{8}{3} d_{\text{model}}$ 근처로 잡는 것이 바로 이 조정의 결과입니다.
