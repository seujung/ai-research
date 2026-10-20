---
title: LLM 핵심 기법
tags:
  - llm
  - rope
  - optimizer
  - moe
---

# LLM 핵심 기법

최근 대형 언어 모델(LLM)은 트랜스포머라는 같은 뼈대 위에 몇 가지 공통 기법을 얹습니다.
위치는 회전으로 넣고(RoPE), 학습 길이보다 긴 문맥은 주파수를 고쳐서 늘립니다.
어텐션 점수가 폭주하지 않게 정규화를 넣고, 학습은 AdamW나 Muon으로 합니다.
FFN 층은 전문가 혼합(MoE)으로 바꾸고, 다음 토큰 여러 개를 함께 예측해 생성을 빠르게 합니다.
추론은 프롬프트를 한 번에 처리하는 prefill과 토큰을 하나씩 만드는 decode로 나뉘고, 두 단계는 병목이 다릅니다.
이 교재는 이 기법을 7개 장으로 정리합니다.

각 장은 개념마다 작동 원리(수식), 손으로 따라갈 수 있는 수치 예시, 그림, 짧은 코드를 붙입니다.
모델별 설정값은 기술 보고서와 Hugging Face에 공개된 `config.json` 에서 확인한 값만 씁니다.

!!! info "다른 교재와의 관계"
    - 어텐션 연산 자체(스케일드 내적, GQA, MLA, 희소·선형 어텐션)는 [어텐션 메커니즘](../attention/index.md)에서 다룹니다. 이 교재는 어텐션 바깥에서 모델 전체에 쓰이는 기법을 다룹니다.
    - 사인·코사인 위치 인코딩과 트랜스포머의 기본 구조는 [신경망 기초 8장](../d2l/08-attention-transformer.md)에, SGD와 Adam의 기초는 [신경망 기초 9장](../d2l/09-optimization.md)에 있습니다.
    - 모델별 전체 특징은 트렌드의 LLM Model 글([Gemma](../../trends/posts/2026-10-20-llm-gemma.md), [DeepSeek](../../trends/posts/2026-10-20-llm-deepseek.md), [Qwen](../../trends/posts/2026-10-20-llm-qwen.md), [GLM](../../trends/posts/2026-10-20-llm-glm.md), [Kimi](../../trends/posts/2026-10-20-llm-kimi.md))에 있습니다.

## 목차

| 장 | 다루는 내용 |
| --- | --- |
| [1. RoPE: 회전 위치 인코딩](01-rope.md) | 2차원 회전과 상대 위치 성질, 주파수 배치, 장거리 감쇠, 부분 RoPE, 분리 RoPE, M-RoPE, 구현 방식 |
| [2. 긴 문맥 확장](02-long-context.md) | 학습 길이를 넘으면 깨지는 이유, 위치 보간(PI), NTK-aware, YaRN, 기저 주파수 키우기, DCA |
| [3. 정규화와 로짓 안정화](03-normalization.md) | LayerNorm과 RMSNorm, 앞뒤 정규화, 로짓 소프트 캡핑, QK-Norm, V-Norm, QK-Clip |
| [4. 옵티마이저](optimizer/index.md) | 모델별 옵티마이저와 하이퍼파라미터, [4.1 Adam](optimizer/01-adam.md)(모멘트 추정, 편향 보정), [4.2 AdamW](optimizer/02-adamw.md)(분리된 가중치 감쇠, 학습률 스케줄, 클리핑), [4.3 Muon](optimizer/03-muon.md)(직교화, 뉴턴-슐츠 반복, Muon Split, QK-Clip) |
| [5. 전문가 혼합(MoE)](05-moe.md) | config로 계산한 총·활성 매개변수, 정방향 계산 손계산, 토큰 선택·전문가 선택·그룹 제한 라우팅, 보조 손실·z-loss·용량·편향 조정, 세분화와 공유 전문가, 희소도, 전문가 병렬과 디코드 메모리 |
| [6. 멀티 토큰 예측(MTP)과 추측 디코딩](06-mtp.md) | 다음 토큰 여러 개를 예측하는 학습 목표, MTP 모듈, 추측 디코딩의 수락 과정 |
| [7. Prefill과 Decode](07-prefill-decode.md) | 추론의 두 단계, 계산 한계와 메모리 한계, TTFT·TPOT, 청크 prefill, 접두사 캐싱, prefill·decode 분리 배포 |

## 각 기법을 쓰는 모델

| 기법 | Gemma | DeepSeek | Qwen | GLM | Kimi |
| --- | --- | --- | --- | --- | --- |
| RoPE 기저 주파수 | 로컬층 $10^4$, 전역층 $10^6$ (Gemma 3·4) | $10^4$ (V3·V3.2·V4) | $10^6$ (Qwen3), $10^7$ (Qwen3-Next·3.5) | $10^6$ (GLM-4.5·5), $8 \times 10^6$ (GLM-5.2) | $5 \times 10^4$ (K2) |
| 부분 RoPE | 전역층 25% (Gemma 4 p-RoPE) | RoPE 전용 64차원 (MLA 분리 RoPE) | 25% (Qwen3-Next·3.5) | 50% (GLM-4·4.5), RoPE 전용 64차원 (GLM-5) | RoPE 전용 64차원 (K2, MLA 분리 RoPE) |
| 긴 문맥 확장 | 전역층 선형 보간 8배 (Gemma 3) | YaRN 40배 (V3·V3.2), 16배 (V4-Pro) | YaRN, DCA (Qwen2·2.5) | 단계별 문맥 확장 학습 (GLM-5) | YaRN 32배 (K2) |
| 로짓 안정화 | 소프트 캡핑 (Gemma 2), QK-Norm (Gemma 3), QK-Norm + V-Norm (Gemma 4) | QK RMSNorm (V4) | QK-Norm (Qwen3 이후) | QK-Norm (GLM-4.5), Muon Split (GLM-5) | QK-Clip (K2) |
| 옵티마이저 | 보고서에 명시 없음 | AdamW (V2·V3), Muon + AdamW (V4) | AdamW (Qwen 1세대), 이후 보고서에 명시 없음 | Muon (GLM-4.5), Muon + Muon Split (GLM-5) | MuonClip = Muon + QK-Clip (K2) |
| MoE | 128개 중 8개 + 병렬 밀집 MLP (Gemma 4 26B A4B) | 256개 중 8개 + 공유 1 (V3·V3.2) | 128개 중 8개 (Qwen3-235B), 512개 중 10개 + 공유 1 (Qwen3-Next) | 160개 중 8개 (GLM-4.5), 256개 중 8개 + 공유 1 (GLM-5) | 384개 중 8개 + 공유 1 (K2) |
| MTP | MTP 드래프터 (Gemma 4) | MTP 모듈 1개 (V3 이후) | MTP 층 1개 (Qwen3.5), Qwen3-Next는 모델 카드에 명시 | MTP 층 1개 (GLM-4.5·5) | 없음 (K2, `num_nextn_predict_layers: 0`) |

## 읽는 방법

1장과 2장은 이어지는 내용입니다. 2장은 1장의 주파수 배치를 전제로 합니다.
3장은 [어텐션 메커니즘 1장](../attention/01-scaled-dot-product.md)의 $\sqrt{d_k}$ 스케일을 읽은 뒤 보면 좋습니다.
4장은 [신경망 기초 9장](../d2l/09-optimization.md)의 Adam을 전제로 합니다. 3장의 QK-Clip과도 이어집니다.
5장과 6장은 다른 장과 독립적입니다.

본문의 수치 예시는 모두 NumPy나 PyTorch로 직접 계산해 확인했습니다.
