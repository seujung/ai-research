---
title: 홈
hide:
  - toc
---

<div class="page-header" markdown>

# AI Research

읽고, 구현하고, 기록합니다. 설명만 적는 대신 실행 가능한 코드와 실제 출력을 함께 싣습니다.

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
    <p class="entry__desc">SGD부터 AdamW까지의 계보를 직접 구현해 비교합니다. RMSProp이 최솟값 근처에서 맴도는 이유, Adam의 편향 보정이 없으면 첫 스텝 보폭이 3배가 되는 현상을 수치로 확인합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/07-learning-rate/">경사 하강법과 학습률</a></h3>
    <p class="entry__desc">학습률이 정하는 것은 방향이 아니라 보폭 하나뿐입니다. 이차 함수에서 안정 조건을 유도하고, 경계를 넘는 순간 수렴이 어떻게 무너지는지 실제 수치로 확인합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/06-loss/">손실 함수</a></h3>
    <p class="entry__desc">손실 함수는 모델이 무엇을 최적이라고 여길지를 결정합니다. 이상치 하나가 MSE의 최적해를 어디까지 끌고 가는지, 교차 엔트로피가 왜 최대 우도와 같은지를 다룹니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/04-backprop/">역전파</a></h3>
    <p class="entry__desc">연쇄 법칙을 계산 그래프에 적용해 2층 MLP의 미분식을 유도하고, 수치 미분으로 검증한 뒤 XOR을 학습시킵니다. 검증이 실패했던 사례와 그 원인까지 함께 다룹니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/03-gradients/">그래디언트</a></h3>
    <p class="entry__desc">편미분을 민감도로 읽는 관점에서 출발해 그래디언트, 야코비안, 헤시안을 구분합니다. Softmax 야코비안의 각 행의 합이 0인 이유를 계산으로 확인합니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/02-activations/">활성화 함수</a></h3>
    <p class="entry__desc">시그모이드부터 SwiGLU까지의 계보를 따라가면서, 포화 구간에서 기울기가 사라지는 현상과 softmax의 수치 안정성 문제를 실제 출력으로 보여 줍니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/book-of-llms/01-mlp/">다층 퍼셉트론</a></h3>
    <p class="entry__desc">뉴런 하나의 계산에서 시작해 층과 배치로 확장하고, 2층 MLP의 순전파를 따라가면서 텐서의 형태가 어떻게 바뀌는지 확인합니다.</p>
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
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/build-llm-from-scratch/02-attention/">어텐션</a></h3>
    <p class="entry__desc">스케일드 닷 프로덕트 어텐션에서 √d_k 로 나누는 이유를 분산 계산으로 확인하고, 인과적 마스킹의 구현을 살펴봅니다.</p>
  </li>

  <li class="entry">
    <div class="entry__meta">
      <time datetime="2026-09-23">2026년 9월 23일</time>
      <span class="entry__cat">서적</span>
    </div>
    <h3 class="entry__title"><a href="books/build-llm-from-scratch/01-tokenization/">토크나이제이션</a></h3>
    <p class="entry__desc">바이트 페어 인코딩이 바이트 단위에서 출발하는 이유와, 한국어가 영어보다 토큰을 훨씬 많이 소모하는 현상을 다룹니다.</p>
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
