---
title: 홈
hide:
  - toc
---

<div class="page-header" markdown>

# AI Research

읽고, 유도하고, 기록합니다. 결론만 적는 대신 수식을 끝까지 따라가고 그림으로 확인합니다.

</div>

<nav class="filter-row" aria-label="분류">
  <span class="chip chip--active">전체</span>
  <a class="chip" href="trends/">트렌드</a>
  <a class="chip" href="books/">서적</a>
  <a class="chip" href="math/">수학</a>
</nav>

<ul class="entry-list">

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/05-optimizer/">옵티마이저</a></h3>
    <p class="entry__desc">SGD부터 AdamW까지 각 규칙이 푸는 문제와 새로 만드는 문제를 분석합니다. RMSProp이 최솟값 근처에서 맴도는 이유와, Adam의 편향 보정이 없으면 첫 스텝 보폭이 3.16배가 되는 이유를 유도합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/07-learning-rate/">경사 하강법과 학습률</a></h3>
    <p class="entry__desc">학습률이 정하는 것은 방향이 아니라 보폭 하나뿐입니다. 안정 조건과 최적 학습률을 유도하고, 미니배치 잡음이 학습률 스케줄을 필요하게 만드는 이유를 계산합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/06-loss/">손실 함수</a></h3>
    <p class="entry__desc">손실 함수는 모델이 무엇을 최적이라고 여길지를 결정합니다. MSE와 MAE의 최적해가 평균과 중앙값인 이유, 교차 엔트로피와 KL 발산의 관계, 라벨 스무딩의 효과를 유도합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/04-backprop/">역전파</a></h3>
    <p class="entry__desc">역방향 계산이 효율적인 이유를 비용으로 설명하고 2층 MLP의 그래디언트를 유도합니다. ReLU의 꺾인 점에서 검증이 실패하는 원인과, 선형 분류기가 XOR에서 ln 2 에 멈추는 증명을 다룹니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/03-gradients/">그래디언트</a></h3>
    <p class="entry__desc">미분을 민감도로 읽는 관점에서 출발해 야코비안과 헤시안을 정의하고, 신경망 한 층과 softmax + 교차 엔트로피의 그래디언트를 단계별로 유도합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/02-activations/">활성화 함수</a></h3>
    <p class="entry__desc">비선형성이 반드시 필요한 이유를 증명하고, 시그모이드부터 SwiGLU까지 기울기 소실, softmax의 온도와 수치 안정성, 게이트의 작동 방식을 분석합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/01-mlp/">다층 퍼셉트론</a></h3>
    <p class="entry__desc">뉴런을 초평면으로 해석하고 층과 배치로 확장합니다. 은닉층이 공간을 접어 XOR을 푸는 과정과, ReLU의 합이 곡선을 근사하는 원리를 따라갑니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">수학</span>
    </div>
    <h3 class="entry__title"><a href="math/linear-algebra/svd/">특이값 분해</a></h3>
    <p class="entry__desc">에카르트-영 정리로 저랭크 근사의 오차를 정량화하고, 이것이 LoRA 같은 기법의 전제와 어떻게 연결되는지 정리합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">수학</span>
    </div>
    <h3 class="entry__title"><a href="math/optimization/">최적화</a></h3>
    <p class="entry__desc">Adam의 갱신 규칙을 단계별로 적고, 1차와 2차 모멘트를 0으로 초기화했을 때 생기는 치우침을 편향 보정 항이 어떻게 제거하는지 설명합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">수학</span>
    </div>
    <h3 class="entry__title"><a href="math/probability/">확률론</a></h3>
    <p class="entry__desc">쿨백-라이블러 발산을 교차 엔트로피와 엔트로피로 분해해서, 언어 모델의 손실 함수가 왜 교차 엔트로피인지를 유도합니다.</p>
  </li>


  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">트렌드</span>
    </div>
    <h3 class="entry__title"><a href="trends/2026/09/23/site-launch/">기록을 남기는 공간을 열었습니다</a></h3>
    <p class="entry__desc">세 섹션을 나눈 기준과, 글을 쓸 때 지키려는 규칙을 정리한 첫 기록입니다.</p>
  </li>

</ul>
