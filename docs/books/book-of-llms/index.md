---
title: 신경망 기초
tags:
  - books
  - llm
  - neural-network
---

# 신경망 기초

다층 퍼셉트론부터 학습률까지, LLM을 이해하는 데 필요한 신경망의 기본기를 정리합니다.
개념마다 직접 실행한 Python 코드와 그 결과를 함께 실어서, 설명과 실제 동작이
어긋나지 않도록 했습니다.

## 정리 현황

- [x] [1. 다층 퍼셉트론](01-mlp.md)
- [x] [2. 활성화 함수](02-activations.md)
- [x] [3. 그래디언트](03-gradients.md)
- [x] [4. 역전파](04-backprop.md)
- [x] [5. 옵티마이저](05-optimizer.md)
- [x] [6. 손실 함수](06-loss.md)
- [x] [7. 경사 하강법과 학습률](07-learning-rate.md)

## 코드 실행 환경

본문의 모든 코드는 NumPy만 사용합니다. 딥러닝 프레임워크가 감추고 있는 계산을
직접 드러내는 것이 목적이기 때문에, 자동 미분에 의존하지 않고 손으로 미분식을 구현합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

문서에 실린 실행 결과는 모두 아래 환경에서 실제로 실행한 출력을 그대로 옮긴 것입니다.

| 항목 | 값 |
| --- | --- |
| NumPy | 2.0.2 |
| 난수 생성기 | `np.random.default_rng(seed)` 로 시드를 고정 |

!!! warning "난수 시드에 대하여"

    `np.random.default_rng` 는 시드를 고정하면 플랫폼과 무관하게 같은 값을 냅니다.
    다만 부동소수점 연산 순서에 따라 마지막 자리는 달라질 수 있으므로,
    출력의 소수점 끝자리가 문서와 완전히 같지 않더라도 정상입니다.
