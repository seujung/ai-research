---
title: 그래디언트
tags:
  - books
  - llm
  - neural-network
  - calculus
---

# 3. 그래디언트

## 편미분은 민감도입니다

어떤 변수에 대한 편미분은 **그 변수를 아주 조금 바꿨을 때 전체 식이 얼마나 변하는지**를
알려 줍니다. 값 자체보다 이 해석이 훨씬 중요합니다.

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

$\partial f / \partial x = -4$ 라면, $x$ 를 1만큼 키울 때 $f$ 가 약 4만큼 줄어든다는 뜻입니다.
학습이란 결국 손실을 줄이는 방향으로 파라미터를 움직이는 일이므로,
각 파라미터의 민감도를 아는 것이 곧 학습의 전제 조건이 됩니다.

## 그래디언트, 야코비안, 헤시안

세 가지는 입력과 출력의 개수에 따라 구분됩니다.

| 이름 | 입력 | 출력 | 형태 | 담고 있는 것 |
| --- | --- | --- | --- | --- |
| 그래디언트 $\nabla f$ | $n$ | 1 | $(n,)$ | 1차 편미분 |
| 야코비안 $J$ | $n$ | $m$ | $(m, n)$ | 1차 편미분 |
| 헤시안 $H$ | $n$ | 1 | $(n, n)$ | 2차 편미분 |

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix},
\qquad
J_{ij} = \frac{\partial f_i}{\partial x_j},
\qquad
H_{ij} = \frac{\partial^{2} f}{\partial x_i \partial x_j}
$$

신경망의 손실 함수는 출력이 스칼라 하나이므로, 파라미터에 대해서는 그래디언트를 씁니다.
반면 층과 층 사이는 벡터를 벡터로 보내는 함수이므로 야코비안이 등장합니다.
역전파는 이 야코비안들을 명시적으로 만들지 않고 벡터-야코비안 곱만 연쇄적으로
계산하는 기법입니다. 야코비안을 실제로 만들면 메모리가 감당되지 않기 때문입니다.

## 수치 미분으로 검증하기

해석적으로 구한 미분식이 맞는지 확인할 때는 중앙 차분을 씁니다.

$$
\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}
$$

한쪽 차분 $\frac{f(x+h)-f(x)}{h}$ 의 오차가 $O(h)$ 인 데 비해,
중앙 차분은 $O(h^{2})$ 이므로 훨씬 정확합니다.
$h$ 는 보통 $10^{-5}$ 에서 $10^{-6}$ 사이를 씁니다. 더 작게 잡으면 오히려
부동소수점 반올림 오차가 커져서 정확도가 떨어집니다.

## 실행 코드

```python title="grad.py"
import numpy as np
np.set_printoptions(precision=6, suppress=True)

# --- 1. 편미분이 곧 민감도입니다 ---
def f(x, y, z):
    return (x + y) * z

x, y, z = -2.0, 5.0, -4.0
print("[편미분 = 민감도]")
print(f"  f({x}, {y}, {z}) = {f(x,y,z)}")
print(f"  해석적: df/dx = z = {z}, df/dy = z = {z}, df/dz = x+y = {x+y}")

h = 1e-5
print(f"  수치적: df/dx = {(f(x+h,y,z)-f(x-h,y,z))/(2*h):.6f}")
print(f"          df/dz = {(f(x,y,z+h)-f(x,y,z-h))/(2*h):.6f}")
print(f"  x를 1만큼 키우면 f는 약 {z}만큼 변합니다. 이것이 '민감도'의 의미입니다.")

# --- 2. 그래디언트: 편미분을 모은 벡터 ---
def g(v):
    return v[0]**2 + 3*v[0]*v[1] + v[1]**3

def numeric_grad(fn, v, h=1e-5):
    out = np.zeros_like(v)
    for i in range(v.size):
        e = np.zeros_like(v); e[i] = h
        out[i] = (fn(v+e) - fn(v-e)) / (2*h)
    return out

v = np.array([1.0, 2.0])
analytic = np.array([2*v[0] + 3*v[1], 3*v[0] + 3*v[1]**2])
print("\n[그래디언트]  g(x,y) = x^2 + 3xy + y^3")
print("  해석적 =", analytic)
print("  수치적 =", numeric_grad(g, v))

# --- 3. 야코비안: 출력이 여러 개인 함수 ---
def softmax(a):
    e = np.exp(a - a.max())
    return e / e.sum()

def softmax_jacobian(a):
    p = softmax(a)
    return np.diag(p) - np.outer(p, p)

a = np.array([1.0, 2.0, 0.5])
J = softmax_jacobian(a)
print("\n[야코비안]  softmax: 입력 3개 -> 출력 3개이므로 3x3 행렬입니다")
print("  p =", softmax(a))
print("  J =\n", J)

J_num = np.zeros((3, 3))
for j in range(3):
    e = np.zeros(3); e[j] = 1e-6
    J_num[:, j] = (softmax(a+e) - softmax(a-e)) / (2e-6)
print("  수치적 야코비안과 일치하는가:", np.allclose(J, J_num, atol=1e-7))
print("  각 행의 합이 0인 이유는 확률의 총합이 항상 1로 고정되기 때문입니다:",
      np.round(J.sum(axis=1), 12))

# --- 4. 헤시안: 출력이 스칼라인 함수의 2차 편미분 ---
def hessian_numeric(fn, v, h=1e-4):
    n = v.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = (fn(v+ei+ej) - fn(v+ei-ej) - fn(v-ei+ej) + fn(v-ei-ej)) / (4*h*h)
    return H

H_analytic = np.array([[2.0, 3.0], [3.0, 6*v[1]]])
print("\n[헤시안]  g의 2차 편미분 행렬")
print("  해석적 =\n", H_analytic)
print("  수치적 =\n", np.round(hessian_numeric(g, v), 4))

def hessian_of_g(v):
    return np.array([[2.0, 3.0], [3.0, 6*v[1]]])

print("\n[헤시안의 고유값으로 곡률의 방향을 읽습니다]")
for point in [np.array([1.0, 2.0]), np.array([1.0, -1.0])]:
    eig = np.linalg.eigvalsh(hessian_of_g(point))
    shape = "모두 양수 -> 아래로 볼록" if (eig > 0).all() else "부호가 섞임 -> 안장점"
    print(f"  점 {point}: 고유값 {np.round(eig, 4)}  ({shape})")
```

## 실행 결과

```text
[편미분 = 민감도]
  f(-2.0, 5.0, -4.0) = -12.0
  해석적: df/dx = z = -4.0, df/dy = z = -4.0, df/dz = x+y = 3.0
  수치적: df/dx = -4.000000
          df/dz = 3.000000
  x를 1만큼 키우면 f는 약 -4.0만큼 변합니다. 이것이 '민감도'의 의미입니다.

[그래디언트]  g(x,y) = x^2 + 3xy + y^3
  해석적 = [ 8. 15.]
  수치적 = [ 8. 15.]

[야코비안]  softmax: 입력 3개 -> 출력 3개이므로 3x3 행렬입니다
  p = [0.231224 0.628532 0.140244]
  J =
 [[ 0.177759 -0.145332 -0.032428]
 [-0.145332  0.23348  -0.088148]
 [-0.032428 -0.088148  0.120576]]
  수치적 야코비안과 일치하는가: True
  각 행의 합이 0인 이유는 확률의 총합이 항상 1로 고정되기 때문입니다: [0. 0. 0.]

[헤시안]  g의 2차 편미분 행렬
  해석적 =
 [[ 2.  3.]
 [ 3. 12.]]
  수치적 =
 [[ 2.  3.]
 [ 3. 12.]]

[헤시안의 고유값으로 곡률의 방향을 읽습니다]
  점 [1. 2.]: 고유값 [ 1.169 12.831]  (모두 양수 -> 아래로 볼록)
  점 [ 1. -1.]: 고유값 [-7.  3.]  (부호가 섞임 -> 안장점)
```

## 결과에서 짚어 볼 점

### Softmax 야코비안의 각 행의 합이 0입니다

$$
J = \mathrm{diag}(p) - p p^{\top}, \qquad
J_{ij} = p_i(\delta_{ij} - p_j)
$$

각 행의 합을 직접 계산해 보면 이유가 드러납니다.

$$
\sum_{j} J_{ij} = p_i \sum_j \delta_{ij} - p_i \sum_j p_j = p_i - p_i = 1 \cdot p_i - p_i \cdot 1 = 0
$$

Softmax의 출력은 항상 합이 1이라는 제약을 받습니다. 어느 방향으로 입력을 흔들어도
확률의 총합은 변하지 않아야 하므로, 변화량의 합이 0이 되는 것은 필연적인 결과입니다.
대각 성분이 양수이고 비대각 성분이 음수인 것도 같은 맥락입니다.
한 클래스의 확률이 올라가면 나머지 클래스의 확률은 반드시 내려갑니다.

### 헤시안의 고유값이 곡률의 방향을 알려 줍니다

같은 함수라도 위치에 따라 헤시안이 달라집니다. $g(x, y) = x^2 + 3xy + y^3$ 의
헤시안은 $\begin{bmatrix} 2 & 3 \\ 3 & 6y \end{bmatrix}$ 이므로 $y$ 값에 의존합니다.

- $(1, 2)$ 에서는 고유값이 $1.169$ 와 $12.831$ 로 모두 양수입니다. 모든 방향으로 위로 휘어 있습니다.
- $(1, -1)$ 에서는 $-7$ 과 $3$ 으로 부호가 섞입니다. 한 방향으로는 올라가고 다른 방향으로는 내려가는 안장점입니다.

고차원 손실 표면에서는 **극솟값보다 안장점이 압도적으로 많습니다.**
$n$ 차원에서 모든 고유값이 우연히 양수일 확률은 차원이 커질수록 급격히 작아지기 때문입니다.
경사 하강법이 안장점 근처에서 느려지는 현상, 그리고 모멘텀이 이를 빠져나오는 데
도움이 되는 이유가 여기에 있습니다.

### 두 고유값의 비율이 학습률을 제약합니다

$(1, 2)$ 에서 고유값의 비율은 $12.831 / 1.169 \approx 11$ 입니다.
이 값을 조건수(condition number)라고 부릅니다. 경사 하강법의 학습률은 가장 큰 고유값에
맞춰서 정해야 발산하지 않는데, 그렇게 잡으면 가장 작은 고유값 방향으로는
11배 느리게 움직이게 됩니다. 적응적 학습률을 쓰는 옵티마이저가 필요한 근본적인 이유입니다.

## 층 단위로 그래디언트를 계산하는 규칙

지금까지는 입력이 몇 개뿐인 함수를 다뤘습니다. 신경망에서는 입력이 행렬이고
중간 결과도 행렬입니다. 이때 그래디언트를 실제로 어떻게 계산하는지 정리합니다.

### 야코비안은 만들지 않습니다

층과 층 사이는 벡터를 벡터로 보내는 함수이므로 원칙적으로는 야코비안이 필요합니다.
그런데 손실은 **스칼라 하나**이므로, 우리가 실제로 원하는 것은 야코비안 전체가 아니라
**손실의 그래디언트에 야코비안을 곱한 결과 하나**입니다.

$$
\underbrace{\frac{\partial L}{\partial x}}_{\text{벡터}} =
\underbrace{\frac{\partial L}{\partial z}}_{\text{벡터}} \cdot
\underbrace{\frac{\partial z}{\partial x}}_{\text{야코비안}}
$$

이 곱을 **벡터-야코비안 곱**이라고 부릅니다. 핵심은 야코비안을 만들지 않고도
이 곱을 직접 계산할 수 있다는 점입니다. 역전파가 성립하는 근거가 바로 이것입니다.

### 규칙 모음

$\delta_Z = \partial L / \partial Z$ 를 "위에서 내려온 그래디언트"라고 부르겠습니다.

| 순전파 | 역전파 | 형태 |
| --- | --- | --- |
| $Z = XW + b$ | $\delta_X = \delta_Z W^\top$ | $(N, D) = (N, H)(H, D)$ |
| | $\delta_W = X^\top \delta_Z$ | $(D, H) = (D, N)(N, H)$ |
| | $\delta_b = \sum_n \delta_Z[n]$ | $(H,)$ |
| $A = f(Z)$ (원소별) | $\delta_Z = \delta_A \odot f'(Z)$ | 형태 그대로 |
| $P = \mathrm{softmax}(Z)$, $L = \mathrm{CE}$ | $\delta_Z = \frac{1}{N}(P - Y_{\text{onehot}})$ | 형태 그대로 |

두 가지 원리로 정리됩니다.

- **행렬 곱은 역전파에서 전치와 함께 반대쪽에 곱해집니다.**
- **순전파의 브로드캐스팅은 역전파에서 합이 됩니다.** $b$ 가 $N$ 개 행에 복제되었으므로,
  그 영향도 $N$ 개 행에서 모아야 합니다.

### 실행 코드

```python title="matcalc.py"
import numpy as np
rng = np.random.default_rng(3)

N, D, H = 4, 3, 5
X = rng.normal(0, 1, (N, D))
W = rng.normal(0, 1, (D, H))
b = rng.normal(0, 1, H)
C = rng.normal(0, 1, (N, H))          # 손실을 스칼라로 만들기 위한 임의의 가중치

def numeric_grad(fn, arr, h=1e-6):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; o = arr[i]
        arr[i] = o + h; fp = fn()
        arr[i] = o - h; fm = fn()
        arr[i] = o
        g[i] = (fp - fm) / (2 * h); it.iternext()
    return g

def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)

print("[규칙 1] 선형 계층  Z = XW + b,  L = sum(C * Z)")
dZ = C                                 # 이 손실에서는 dL/dZ 가 곧 C 입니다
dX_an, dW_an, db_an = dZ @ W.T, X.T @ dZ, dZ.sum(axis=0)
L = lambda: float((C * (X @ W + b)).sum())
print(f"  dL/dX = dZ @ W.T   상대오차 {rel(dX_an, numeric_grad(L, X)):.3e}")
print(f"  dL/dW = X.T @ dZ   상대오차 {rel(dW_an, numeric_grad(L, W)):.3e}")
print(f"  dL/db = dZ.sum(0)  상대오차 {rel(db_an, numeric_grad(L, b)):.3e}")
```

### 실행 결과

```text
[규칙 1] 선형 계층  Z = XW + b,  L = sum(C * Z)
  dL/dX = dZ @ W.T   상대오차 3.091e-10
  dL/dW = X.T @ dZ   상대오차 1.479e-10
  dL/db = dZ.sum(0)  상대오차 3.238e-10
  형태 확인: X(4, 3) W(3, 5) -> Z(4, 5),  dW(3, 5) == W(3, 5)

[규칙 2] 원소별 활성화  A = tanh(Z),  L = sum(C * A)
  dL/dZ = dA * (1 - tanh(Z)^2)   상대오차 6.127e-10
  dL/dX (연쇄 적용)              상대오차 4.391e-10

[규칙 3] 브로드캐스팅은 역전파에서 합이 됩니다
  순전파: b(5,) 가 4개 행에 복제되어 더해집니다
  역전파: dL/db 는 배치 축으로 합을 취해 (5,) 로 돌아옵니다
  dL/db = [ 2.206486 -1.261885 -0.439345  0.68828  -1.647919]
  dZ 의 열별 합 = [ 2.206486 -1.261885 -0.439345  0.68828  -1.647919]

[규칙 4] softmax + 교차 엔트로피
  dL/dlogits = (P - onehot)/N    상대오차 6.171e-10
  각 행의 합 = [ 0. -0.  0. -0.]  (확률 총합이 1로 고정되기 때문입니다)

[규칙 5] 야코비안을 만들지 않고 벡터-야코비안 곱만 계산합니다
  선형 계층의 야코비안 dZ/dX 는 (20, 12) = 240개 원소입니다
  그러나 실제로 필요한 것은 dZ @ W.T 뿐이며, 이는 4x5 @ 5x3 곱 한 번입니다
  d_model=4096, 배치=1024 인 경우 야코비안은 1.759e+13개 원소로,
  float32 기준 약 70 TB 입니다. 만들 수 없습니다.
  반면 벡터-야코비안 곱에 필요한 dZ 는 1024x4096 = 17 MB 뿐입니다.
```

### 결과에서 짚어 볼 점

**네 규칙 모두 수치 미분과 $10^{-10}$ 수준에서 일치합니다.** 손으로 유도한 식이
맞다는 것을 확인하는 가장 확실한 방법이며, 새 층을 직접 구현할 때마다
이 검증을 거치는 습관이 도움이 됩니다.

**규칙 3은 브로드캐스팅과 합이 서로 짝이라는 점을 보여 줍니다.** `dL/db` 가
`dZ` 의 열별 합과 정확히 같습니다. 순전파에서 값 하나를 여러 곳에 복제했다면,
역전파에서는 그 여러 곳에서 온 기여를 모두 더해야 합니다. 반대로 순전파에서
합을 취했다면 역전파에서는 복제가 됩니다.

**규칙 5가 역전파의 존재 이유입니다.** 선형 계층 하나의 야코비안을 실제로 만들면
$d_{\text{model}} = 4096$, 배치 1024 기준으로 약 70 TB가 필요합니다.
그런데 정작 필요한 것은 17 MB짜리 행렬 하나이고, 그것도 행렬 곱 한 번으로 얻습니다.

**중간 계산을 건너뛸 수 있기 때문에 규모가 큰 모델의 학습이 가능해집니다.**
야코비안을 만들어야만 했다면 오늘날의 언어 모델은 존재할 수 없었습니다.

### 형태로 식을 복원하는 방법

미분식을 외우기보다 형태를 맞추는 편이 실수가 적습니다.
$\delta_W$ 는 $W$ 와 같은 $(D, H)$ 여야 하는데, 재료는 $X$ 가 $(N, D)$ 이고
$\delta_Z$ 가 $(N, H)$ 입니다. 이 둘로 $(D, H)$ 를 만드는 조합은
$X^\top \delta_Z$ 하나뿐입니다.

그래디언트는 **항상 대응하는 변수와 같은 형태**를 가집니다.
이 사실 하나만 기억하면 대부분의 식을 그 자리에서 복원할 수 있습니다.

## 다음 장

이 규칙들을 실제 2층 신경망에 연쇄적으로 적용하는 것이
[다음 장의 역전파](04-backprop.md)입니다.
