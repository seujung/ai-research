# 양자역학 교재 수식 전개 보강 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/physics/quantum-mechanics/` 의 12개 장에서 생략된 수식 전개를 본문에 펼치고, 개념 그림을 장마다 하나씩 더해 초보자가 각 줄을 따라갈 수 있게 합니다.

**Architecture:** 장 단위로 작업합니다. 각 장에서 조사로 확인된 상위 3~5개 생략 지점을 펼치고, 그 장의 가장 추상적인 개념 하나를 도해로 보강합니다. 수치는 `sympy`·`numpy` 로 먼저 검증한 뒤 본문에 씁니다. 장마다 `mkdocs build --strict` 와 한국어 린터로 확인하고 커밋합니다.

**Tech Stack:** Markdown + MathJax(pymdownx.arithmatex), MkDocs Material, matplotlib(PNG 생성), sympy/numpy(수치 검증)

**Spec:** `docs/superpowers/specs/2026-11-23-qm-derivation-detail-design.md`

## Global Constraints

- 문체는 합쇼체(`-ㅂ니다`)로 통일합니다. 기존 장�� 합쇼체이므로 그대로 따릅니다.
- 한 문장에 생각 하나. 지시문 15어절, 설명문 20어절 이하.
- 한 개념에 한 용어. "전개", "정리", "계산"을 같은 동작에 섞지 않습니다.
- 수식은 `$$ ... $$` 블록과 인라인 `$...$` 만 씁니다. MathJax 설정이 로드하는 패키지는 `boldsymbol` 뿐이므로 `\boxed`, `\underbrace`, `\dfrac`, `\begin{cases}` 는 기존 장에서 이미 쓰는 범위 안에서만 씁니다.
- 새 매크로를 정의하지 않습니다. `mathjax.js` 를 고치지 않습니다.
- 기존 수치, 결론 문장, 양태 표현(`-ㄹ 수 있습니다`), 각주 링크를 바꾸지 않습니다. 전개를 더할 뿐입니다.
- 그림은 프로젝트 `.venv` 에 matplotlib 이 없으므로 별도 venv `/tmp/qmfig` 에서 생성합니다. 없으면 `python3 -m venv /tmp/qmfig && /tmp/qmfig/bin/pip install matplotlib numpy scipy sympy` 로 다시 만듭니다.
- 그림 공통 설정: 한글 글꼴 `Apple SD Gothic Neo`, `figsize` 폭 18(inch)·`dpi=100`, 흰 배경, `facecolor='white'`, 저장 시 `bbox_inches='tight'`, 파일 크기 100~250 KB. 색은 기존 그림과 맞춥니다: 파랑 `#2a76d2`, 주황 `#e7832b`, 초록 `#308e58`, 빨강 `#a03c42`, 회색 `#8a8a8a`.
- 로그 축 눈금은 `FuncFormatter` 로 `$10^{n}$` 형식을 직접 지정합니다. 유니코드 마이너스가 글꼴에 없어 깨집니다.
- 그림 생성 스크립트는 저장소에 넣지 않습니다. PNG 만 `docs/physics/quantum-mechanics/images/` 에 둡니다.
- figure 블록은 기존 형식을 그대로 씁니다:

```markdown
<figure markdown>
![대체 텍스트](images/NN-slug.png)
<figcaption markdown>
<span class="fig-title"><b>Figure K.</b> 제목</span>

설명 두세 문장. 무엇을 보는지와 수치를 적습니다.
</figcaption>
</figure>
```

- 새 그림을 넣으면 그 장의 뒤쪽 Figure 번호를 1씩 올립니다. 장 안 번호만 셉니다.
- 확인 명령은 저장소 루트에서 실행합니다. `cd /Users/digit82/repo/ai-research`.
- 커밋 메시지는 저장소 관례를 따르고 다음 줄로 끝냅니다:

```
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

## Review Focus

- **수식 오류**: 중간 식을 새로 쓰므로 부호, 계수, 지수가 틀릴 수 있습니다. 모든 새 결과는 `sympy` 로 기존 결과와 같은지 확인합니다. 가장 위험한 곳은 2장 장벽 투과율, 5장 통계 분포, 6장 상대론 보정, 8장 가모프 적분입니다.
- **기존 수치와의 불일치**: 전개를 펼치다 기존 표의 숫자와 다른 값이 나오면 기존 숫자를 먼저 의심합니다. 확인 없이 표를 고치지 않습니다.
- **한국어 문체 이탈**: 새로 쓴 문장이 해라체나 번역투로 흘러갈 수 있습니다. 장 전체를 다시 읽어 확인합니다.
- **그림 한글 깨짐**: 글꼴이 없으면 한글이 네모나 `¤` 로 나옵니다. PNG 를 직접 열어 확인합니다. 특히 마이너스 부호와 로그 눈금입니다.
- **링크·앵커 깨짐**: 소제목을 바꾸면 기존 `#앵커` 링크가 깨집니다. 소제목 문구를 바꾸지 않는 것이 기본입니다.

---

### Task 0: 그림 도구와 검증 명령 준비

**Files:**
- Create: (저장소 밖) `/tmp/qmfig/`
- Create: (저장소 밖) `/tmp/qmwork/`

**Interfaces:**
- Produces: `/tmp/qmfig/bin/python` (matplotlib 3.9, numpy, scipy, sympy 설치), `/tmp/qmwork/` (그림·검증 스크립트 임시 보관)

- [ ] **Step 1: 그림 venv 를 만듭니다**

Run:
```bash
python3 -m venv /tmp/qmfig && /tmp/qmfig/bin/pip install -q matplotlib numpy scipy sympy
/tmp/qmfig/bin/python -c "import matplotlib; print(matplotlib.__version__)"
```
Expected: `3.9.4` (또는 3.9 이후)

- [ ] **Step 2: 작업 디렉터리를 만듭니다**

Run: `mkdir -p /tmp/qmwork && ls -d /tmp/qmwork`
Expected: `/tmp/qmwork`

- [ ] **Step 3: 공용 그림 설정 파일을 만듭니다**

`/tmp/qmwork/qmstyle.py` 를 만듭니다. 이 파일은 모든 그림 스크립트가 가져다 씁니다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "blue": "#2a76d2", "orange": "#e7832b", "green": "#308e58",
    "red": "#a03c42", "gray": "#8a8a8a",
}

def setup():
    plt.rcParams.update({
        "font.family": "Apple SD Gothic Neo",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#cacaca",
        "axes.labelcolor": "#3a3a3a",
        "xtick.color": "#6a6a6a",
        "ytick.color": "#6a6a6a",
        "axes.grid": False,
        "font.size": 13,
    })

def save(fig, name):
    out = "/Users/digit82/repo/ai-research/docs/physics/quantum-mechanics/images/" + name
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
```

- [ ] **Step 4: 한글 글꼴이 잡히는지 확인합니다**

Run:
```bash
cd /tmp/qmwork && /tmp/qmfig/bin/python -c "
import qmstyle, matplotlib.pyplot as plt
qmstyle.setup()
fig, ax = plt.subplots(figsize=(4,2)); ax.text(0.5,0.5,'확률 흐름 −1',ha='center',fontsize=16); ax.axis('off')
qmstyle.save(fig, '_fontcheck.png')"
```
Expected: `saved .../_fontcheck.png`. 파일을 열어 `확률 흐름 −1` 이 깨지지 않고 보이면 통과입니다. 확인 뒤 `_fontcheck.png` 를 지웁니다.

- [ ] **Step 5: 빌드와 린터 명령이 동작하는지 확인합니다**

Run:
```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/01-wave-function.md | tail -3
```
Expected: `build=0`, 린터가 요약 줄을 출력합니다. 린터가 없으면 이 단계를 건너뛰고 이후 검사에서 뺍니다.

- [ ] **Step 6: 커밋하지 않습니다**

Task 0 은 저장소를 바꾸지 않습니다. 커밋할 파일이 없습니다.

---

### Task 1: 1장 파동 함수

**Files:**
- Modify: `docs/physics/quantum-mechanics/01-wave-function.md` (242, 258, 262, 288–311, 333–338, 371, 388, 408–412줄 부근)
- Create: `docs/physics/quantum-mechanics/images/01-probability-current.png`

**Interfaces:**
- Consumes: `/tmp/qmwork/qmstyle.py` 의 `setup()`, `save(fig, name)`, `COLORS`
- Produces: 1장 Figure 4 를 `01-probability-current.png` 로 새로 넣고, 기존 Figure 4~6 을 5~7 로 올림

- [ ] **Step 1: 규격화 적분의 이항 전개를 펼칩니다 (242줄)**

`x^2(a-x)^2` 를 전개하는 단계를 넣습니다. 기존 `A^2(a^5/3 - 2a^5/4 + a^5/5)` 결과는 유지합니다.

```markdown
먼저 $x^2 (a - x)^2$ 를 전개합니다.

$$
x^2(a-x)^2 = x^2(a^2 - 2ax + x^2) = a^2 x^2 - 2a x^3 + x^4
$$

항마다 적분합니다. $\int_0^a x^n dx = a^{n+1}/(n+1)$ 입니다. $n = 2, 3, 4$ 를 넣으면 $a^3/3$, $a^4/4$, $a^5/5$ 입니다.
```

그다음 기존 식을 둡니다. 기존 식의 `a^5/3`, `2a^5/4`, `a^5/5` 가 이 적분에서 나온 값임을 밝히는 한 문장을 뒤에 붙입니다.

- [ ] **Step 2: 기댓값 적분의 거듭제곱 값을 펼칩니다 (258, 262줄)**

`x \cdot x^2(a-x)^2 = a^2x^3 - 2ax^4 + x^5` 와 `x^4(a-x)^2 = a^2x^4 - 2ax^5 + x^6` 을 보이고, 각 적분값 `a^6/60`, `a^7/105` 가 어디서 나오는지 씁니다.

```markdown
$x \cdot x^2(a-x)^2 = a^2 x^3 - 2a x^4 + x^5$ 이므로
$\int_0^a = a^6/4 - 2a^6/5 + a^6/6 = a^6/60$ 입니다.
```

- [ ] **Step 3: 규격화 보존 증명의 곱의 미분과 전미분 묶기를 펼칩니다 (300~305줄)**

"전미분 꼴로 묶입니다" 앞에 역곱 규칙을 확인하는 한 줄을 넣습니다.

```markdown
$\Psi^* \Psi'' - (\Psi^*)'' \Psi = \frac{\partial}{\partial x}(\Psi^* \Psi' - (\Psi^*)' \Psi)$ 는
곱의 미분 $\partial_x(\Psi^*\Psi') = (\Psi^*)'\Psi' + \Psi^*\Psi''$ 에서 두 항의 $(\Psi^*)'\Psi'$ 가 지워지는 것입니다.
```

- [ ] **Step 4: d⟨x⟩/dt 의 부분 적분 두 번을 펼칩니다 (333~338줄)**

기존 문장 "앞 절의 연속 방정식에 부분 적분을 두 번 적용하면 다음 식이 나옵니다"를 다음 전개로 바꿉니다. 최종 식은 기존 식과 같습니다.

```markdown
연속 방정식 $\partial_t |\Psi|^2 = -\partial_x J$ 를 넣습니다.

$$
\frac{d\langle x\rangle}{dt} = \int x \left(-\frac{\partial J}{\partial x}\right) dx
$$

부분 적분합니다. 확률 흐름 $J$ 는 $x \to \pm\infty$ 에서 0이므로 $[xJ]$ 경계항이 사라집니다.

$$
= \int J \, dx
$$

$J = \frac{i\hbar}{2m}(\Psi^* \partial_x \Psi - \partial_x \Psi^* \Psi)$ 를 넣습니다.

$$
= \frac{i\hbar}{2m}\int \left(\Psi^* \frac{\partial \Psi}{\partial x} - \frac{\partial \Psi^*}{\partial x}\Psi\right) dx
$$

두 번째 항에 부분 적분을 한 번 더 합니다. 경계항은 다시 0이고, 남는 항은 첫 항과 같은 모양입니다.

$$
= \frac{i\hbar}{m}\int \Psi^* \frac{\partial \Psi}{\partial x} dx
$$

$\frac{i\hbar}{m} = -\frac{1}{m}(-\hbar/i)$ 이므로 부호를 정리하면 다음 식입니다.

$$
\frac{d \langle x \rangle}{dt} = -\frac{i\hbar}{m} \int \Psi^* \frac{\partial \Psi}{\partial x} \, dx
$$
```

- [ ] **Step 5: ⟨p²⟩ 와 ⟨T⟩ 의 부분 적분을 펼칩니다 (371, 388줄)**

- 371줄 앞에 "$\int \Psi^* \Psi'' dx = -\int |\Psi'|^2 dx$ 는 부분 적분에서 경계항이 0이 되기 때문" 한 문장과 식을 넣습니다.
- 388줄에서 `x(a-x)` 의 2차 도함수가 `-2` 인 것과 `∫_0^a x(a-x)dx = a^3/6` 인 것을 한 줄씩 보입니다.

- [ ] **Step 6: 확률 흐름 그림을 만듭니다**

`/tmp/qmwork/fig01.py` 를 쓰고 실행합니다. 가로축 $x$, 세로축 $J(x)$ 로, 오른쪽으로 흐르는 흐름을 화살표로 보여 주고, $[xJ]_{-\infty}^{\infty} = 0$ 이 되는 양 끝을 회색 음영으로 표시합니다.

Run:
```bash
cd /tmp/qmwork && /tmp/qmfig/bin/python fig01.py
```
Expected: `saved .../01-probability-current.png`, 파일 크기 100~250 KB

- [ ] **Step 7: 그림을 본문에 넣고 번호를 올립니다**

4단계 전개 뒤에 figure 블록을 넣습니다. 대체 텍스트는 "확률 흐름과 부분 적분의 경계항"입니다. 캡션은 $J$ 가 흐르는 방향과 경계항이 0이 되는 이유를 설명합니다. 그 뒤 기존 Figure 4, 5, 6 을 5, 6, 7 로 올립니다.

- [ ] **Step 8: ���드와 린터로 확인합니다**

Run:
```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/01-wave-function.md | tail -5
```
Expected: `build=0`. 린터 hard 위반 0. advisory 는 인용문에서 생기는 것만 허용합니다.

- [ ] **Step 9: 커밋합니다**

```bash
git add docs/physics/quantum-mechanics/01-wave-function.md docs/physics/quantum-mechanics/images/01-probability-current.png
git commit -m "Expand derivations in quantum mechanics chapter 1"
```

---

### Task 2: 2장 시간에 무관한 슈뢰딩거 방정식

**Files:**
- Modify: `docs/physics/quantum-mechanics/02-time-independent-schrodinger.md` (151–156, 189–198, 227–236, 245–249, 285, 413–419, 535–542줄 부근)
- Create: `docs/physics/quantum-mechanics/images/02-fourier-coefficient.png`

**Interfaces:**
- Consumes: Task 0 의 `qmstyle.py`
- Produces: 2장 Figure 3 을 새 그림으로 넣고 기존 Figure 3~7 을 4~8 로 올림

- [ ] **Step 1: 푸리에 계수 적분을 두 번 부분 적분해 펼칩니다 (151–156줄)**

"부분 적분을 두 번 하면 닫힌 식이 나옵니다"를 다음 흐름으로 바꿉니다.

1. $\int_0^a x(a-x)\sin(kx)dx$ 에서 $u = x(a-x)$, $dv = \sin(kx)dx$, $k = n\pi/a$ 로 둡니다.
2. 첫 부분 적분 결과가 $[-\frac{x(a-x)}{k}\cos kx]_0^a + \frac{1}{k}\int_0^a (a-2x)\cos kx\,dx$ 임을 보입니다. 경계항은 0입니다.
3. 두 번째 부분 적분으로 $\frac{1}{k}[\frac{a-2x}{k}\sin kx]_0^a + \frac{2}{k^2}\int_0^a \sin kx\,dx$ 를 얻습니다. 첫 항도 0입니다.
4. $\int_0^a \sin kx\,dx = (1-\cos n\pi)/k$ 이므로 $n$ 홀수일 때 $2/k$ 입니다.
5. 모으면 $\frac{4}{k^3} = \frac{4a^3}{n^3\pi^3}$ 입니다. 계수 $\sqrt{2/a}\sqrt{30/a^5}$ 를 곱하면 기존 결과 $\frac{8\sqrt{15}}{(n\pi)^3}$ 입니다. 이 마지막 곱셈을 한 줄로 보입니다.

- [ ] **Step 2: 두 상태 중첩의 교차항을 펼칩니다 (189–198줄)**

$|\psi_1 + \psi_2|^2/\sqrt{2}$ 의 제곱을 전개해 교차항이 $2\psi_1\psi_2\cos\omega t$ 가 되는 과정을 보입니다. 위상 인자 $e^{-iE_1t/\hbar}$, $e^{-iE_2t/\hbar}$ 의 곱에서 $e^{-i(E_2-E_1)t/\hbar}$ 가 나오고, 켤레와 더해 $2\cos\omega t$ 가 되는 두 줄을 넣습니다.

- [ ] **Step 3: ⟨x⟩(t) 의 적분값 −16a/9π² 을 보입니다 (194~198줄)**

$x_{12} = \int_0^a x\psi_1\psi_2 dx = -16a/(9\pi^2)$ 를 3장의 $x_{mn}$ 공식에서 온다고 밝히고, 이 값이 $\cos\omega t$ 계수 $\frac{2}{\sqrt2\cdot\sqrt2}x_{12}$ 에서 나오는 것을 한 줄로 씁니다.

- [ ] **Step 4: 사다리 연산자의 곱 전개를 펼칩니다 (227–236줄)**

$a_\pm$ 의 정의를 다시 쓰고, $a_-a_+$ 를 $x$ 와 $p$ 로 전개합니다.

```markdown
$a_\pm = \frac{1}{\sqrt{2m\hbar\omega}}(m\omega x \mp ip)$ 입니다. 곱을 전개합니다.

$$
a_- a_+ = \frac{1}{2m\hbar\omega}\left[(m\omega x + ip)(m\omega x - ip)\right]
= \frac{1}{2m\hbar\omega}\left[m^2\omega^2 x^2 + p^2 - im\omega(xp - px)\right]
$$

$[x,p] = i\hbar$ 이므로 $xp - px = i\hbar$ 입니다. 마지막 항은 $-im\omega \cdot i\hbar = m\hbar\omega$ 입니다.

$$
= \frac{1}{2m\hbar\omega}\left[m^2\omega^2 x^2 + p^2 + m\hbar\omega\right]
= \frac{1}{\hbar\omega}\left(\frac{p^2}{2m} + \frac{1}{2}m\omega^2x^2\right) + \frac{1}{2}
= \frac{\hat H}{\hbar\omega} + \frac{1}{2}
$$

같은 방법으로 $a_+a_- = \hat H/\hbar\omega - 1/2$ 입니다. 두 식을 빼면 $[a_-, a_+] = 1$ 입니다.
```

- [ ] **Step 5: 바닥 상태 미분 방정식을 풉니다 (245–249줄)**

$(\hbar\frac{d}{dx} + m\omega x)\psi_0 = 0$ 에서 $\frac{d\psi_0}{\psi_0} = -\frac{m\omega}{\hbar}x\,dx$ 로 변수를 분리하고, 적분해 $\ln\psi_0 = -\frac{m\omega}{2\hbar}x^2 + C$ 를 얻는 두 줄을 넣습니다. 규격화 상수 $(\frac{m\omega}{\pi\hbar})^{1/4}$ 는 $\int e^{-m\omega x^2/\hbar}dx = \sqrt{\pi\hbar/m\omega}$ 에서 나온다고 밝힙니다.

- [ ] **Step 6: 장벽 투과율의 경계 조건을 펼칩니다 (535–542줄)**

투과율 공식만 주던 곳에 네 경계 조건을 세우는 과정을 넣습니다. 세 영역의 파동 함수(입사·반사, 감쇠, 투과)를 쓰고, $x=0$ 과 $x=L$ 에서 $\psi$ 와 $\psi'$ 의 연속 조건 네 개를 식으로 보입니다. 이어서 두 식을 나누어 계수를 소거하는 방향을 한 단계 보입니다. 최종 $T(E)$ 식은 기존 식을 그대로 둡니다.

수치 확인은 `sympy` 로 합니다. $E = 0.5\,\mathrm{eV}$, $V_0 = 1\,\mathrm{eV}$, $L = 1\,\mathrm{nm}$, $m = m_e$ 에서 기존 식과 경계 조건을 직접 푼 결과가 같은지 봅니���.

- [ ] **Step 7: 푸리에 계수 그림을 만듭니다**

`/tmp/qmwork/fig02.py`: $x(a-x)\sin(n\pi x/a)$ 를 $n = 1, 2, 3$ 에 대해 그리고, 짝수 $n$ 은 양·음이 상쇄되어 적분이 0이 됨을 음영으로 보입니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig02.py`
Expected: `saved .../02-fourier-coefficient.png`

- [ ] **Step 8: 그림을 넣고 번호를 올립니다**

1단계 전개 뒤에 넣고, 기존 Figure 3~7 을 4~8 로 올립니다.

- [ ] **Step 9: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/02-time-independent-schrodinger.md | tail -5
git add docs/physics/quantum-mechanics/02-time-independent-schrodinger.md docs/physics/quantum-mechanics/images/02-fourier-coefficient.png
git commit -m "Expand derivations in quantum mechanics chapter 2"
```

---

### Task 3: 3장 형식 체계

**Files:**
- Modify: `docs/physics/quantum-mechanics/03-formalism.md` (109–110, 166–173, 256, 301–309, 359–364, 393–398, 417–429줄 부근)
- Create: `docs/physics/quantum-mechanics/images/03-schwarz-geometry.png`

**Interfaces:**
- Consumes: Task 0 의 `qmstyle.py`
- Produces: 3장 Figure 4 를 새 그림으로 넣고 기존 Figure 4~5 를 5~6 으로 올림

- [ ] **Step 1: 3×3 고유값 분해를 펼칩니다 (109–110줄)**

특성 다항식 $\det(C - \lambda I) = (2-\lambda)[(2-\lambda)^2 - 1] - (2-\lambda) = (2-\lambda)[(2-\lambda)^2 - 2]$ 를 보이고, 근 $\lambda = 2$, $2 \pm \sqrt2$ 를 얻습니다. 이어서 $\lambda = 2$ 의 고유벡터를 연립 방정식 $a_1 + a_3 = 0$ 에서 $(1, 0, -1)/\sqrt2$ 로 구하는 두 줄을 넣습니다. 나머지 두 고유벡터는 같은 방법이라고 밝힙니다.

- [ ] **Step 2: x 행렬 성분의 삼각함수 곱을 합으로 바꿉니다 (166–173줄)**

$\sin A\sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$ 를 명시하고, $m \ne n$ 일 때 $x_{mn} = \frac{1}{a}\int_0^a x[\cos\frac{(m-n)\pi x}{a} - \cos\frac{(m+n)\pi x}{a}]dx$ 가 되는 것을 보입니다. 부분 적분 한 번으로 $\int x\cos(kx)dx = \frac{x\sin kx}{k} + \frac{\cos kx}{k^2}$ 의 경계값이 남고, $m+n$ 이 홀수일 때 $-\frac{8amn}{\pi^2(m^2-n^2)^2}$ 이 나오는 흐름을 세 줄로 보입니다.

- [ ] **Step 3: 운동량 공간 파동 함수의 적분을 펼칩니다 (301–309줄)**

"적분을 끝까지 하면"을 다음으로 바꿉니다.

1. $u = pa/\hbar$ 로 치환해 적분 구간을 $[0, \pi]$ 로 바꿉니다.
2. $\sin(\pi x/a)e^{-iux/a}$ 를 지수 함수 두 개로 바꿉니다.
3. $\int_0^\pi e^{i\alpha t}dt = \frac{e^{i\alpha\pi}-1}{i\alpha}$ 를 $\alpha = \pm 1 \mp u$ 에 적용합니다.
4. $\frac{e^{i\alpha\pi}-1}{i\alpha} = e^{i\alpha\pi/2}\frac{2\sin(\alpha\pi/2)}{\alpha}$ 로 정리합니다.
5. 크기 제곱에서 위상이 사라지고 $\cos^2(u/2)/(\pi^2-u^2)^2$ 이 나오는 마지막 한 줄을 보입니다.

최종 $|\Phi(p)|^2$ 식은 기존 식을 유지합니다. 수치는 `sympy` 로 $\int|\Phi|^2dp = 1$ 을 다시 확인합니다.

- [ ] **Step 4: 불확정성 원리의 슈바르츠 대입을 펼칩니다 (359–364줄)**

$f = (A - \langle A\rangle)\Psi$, $g = (B - \langle B\rangle)\Psi$ 로 두고 $\sigma_A^2 = \langle f|f\rangle$, $\sigma_B^2 = \langle g|g\rangle$ 를 밝힙니다. 슈바르츠 부등식 $\langle f|f\rangle\langle g|g\rangle \ge |\langle f|g\rangle|^2$ 에서 $z = \langle f|g\rangle = x + iy$ 로 두고 $|z|^2 \ge y^2 = (\frac{z-z^*}{2i})^2$ 를 씁니다. $z - z^* = \langle [A,B]\rangle$ 이 되는 두 줄을 넣습니다.

- [ ] **Step 5: 슈바르츠 부등식 기하를 그립니다**

`/tmp/qmwork/fig03.py`: 2차원에서 벡터 $f$, $g$ 와 $f$ 에 내린 수선을 그리고, $|z|^2 \ge y^2$ 이 수선 길이의 제곱이라는 것을 보입니다. $y$ 를 빨간 선분으로 강조합니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig03.py`

- [ ] **Step 6: 그림을 넣고 번호를 올립니다**

4단계 전개 뒤에 넣고, 기존 Figure 4~5 를 5~6 으로 올립니다. 새 그림은 Figure 4 입니다.

- [ ] **Step 7: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/03-formalism.md | tail -5
git add docs/physics/quantum-mechanics/03-formalism.md docs/physics/quantum-mechanics/images/03-schwarz-geometry.png
git commit -m "Expand derivations in quantum mechanics chapter 3"
```

- [ ] **Step 8: 여기서 사용자에게 중간 결과를 보여 줍니다**

1~3장을 마치면 사용자에게 바뀐 곳 세 예를 보여 주고 방향이 맞는지 확인합니다. 승인 없이 4장으로 넘어가지 않습니다.

---

### Task 4: 4장 3차원 양자역학

**Files:**
- Modify: `docs/physics/quantum-mechanics/04-three-dimensions.md` (40–49, 56–59, 146–150, 154–166, 250–253, 291–296, 306–312, 469–481줄 부근)
- Create: `docs/physics/quantum-mechanics/images/04-ladder-lz.png`

- [ ] **Step 1: 변수 분리의 나눗셈을 펼칩니다 (40–49줄)**

$\psi = R(r)Y(\theta,\phi)$ 를 넣고 $\frac{1}{R}\frac{d}{dr}(r^2\frac{dR}{dr}) - \frac{2mr^2}{\hbar^2}[V(r)-E] = -\frac{1}{Y}\left[\frac{1}{\sin\theta}\partial_\theta(\sin\theta\partial_\theta Y) + \frac{1}{\sin^2\theta}\partial_\phi^2Y\right]$ 를 얻는 과정을 보입니다. 좌변이 $r$ 만, 우변이 각도만의 함수이므로 둘 다 상수이고, 그 상수를 $l(l+1)$ 로 부르는 것을 명시합니다.

- [ ] **Step 2: 방사 방정식의 무차원화를 펼칩니다 (146–150줄)**

$\kappa = \sqrt{-2mE}/\hbar$, $\rho = \kappa r$, $\rho_0 = \frac{2m e^2}{4\pi\epsilon_0\hbar^2\kappa}$ 를 정의에서 대입해 $u'' = [1 - \rho_0/\rho + l(l+1)/\rho^2]u$ 가 나오는 세 줄을 넣습니다.

- [ ] **Step 3: 멱급수 해법의 대입과 점화식을 펼칩니다 (154–166줄)**

$u = \rho^{l+1}e^{-\rho}v(\rho)$ 를 넣고 $v = \sum c_j\rho^j$ 를 대입해 점화식 $c_{j+1} = \frac{2(j+l+1)-\rho_0}{(j+1)(j+2l+2)}c_j$ 가 나오는 흐름을 보입니다. 급수가 끊기는 조건 $\rho_0 = 2n$ 을 수렴 조건에서 얻는 한 줄을 넣습니다.

- [ ] **Step 4: P(r<a) 의 부분 적분 두 번을 펼칩니다 (250–253줄)**

$\int r^2 e^{-2r/a}dr$ 에서 $u = r^2$, $dv = e^{-2r/a}dr$ 로 두 번 부분 적분해 $\left[-\frac{a}{2}r^2e^{-2r/a}\right]_0^a + a\int r e^{-2r/a}dr$, 이어서 $-\frac{a^2}{2}e^{-2} + \frac{a^2}{4}(1 - e^{-2}\cdot 5)$ 을 거쳐 기존 결과 $1-5e^{-2}$ 에 이르는 식을 보입니다.

- [ ] **Step 5: 각운동량 교환자를 펼칩니다 (291–296줄)**

$L_x = yp_z - zp_y$ 로 쓰고 $[L_x, L_y]$ 를 네 항으로 전개합니다. $[y,p_y] = i\hbar$ 같은 기본 교환자만 남기는 과정을 한 번 완전히 보입니다. 나머지 두 교환자는 같은 방법이라고 밝힙니다.

- [ ] **Step 6: L_z 사다리 그림을 그립니다**

`/tmp/qmwork/fig04.py`: $l = 2$ 의 $m = -2,\dots,2$ 준위를 수평선으로 그리고, $L_\pm$ 이 이웃 준위를 잇는 화살표를 그립니다. 각 준위 옆에 $\sqrt{l(l+1)-m(m\pm1)}$ 계수를 적습니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig04.py`

- [ ] **Step 7: 그림을 넣고 번호를 올립니다**

5단계 뒤에 넣고, 기존 Figure 4~7 을 5~8 로 올립니다.

- [ ] **Step 8: 469~481줄의 스피너 기댓값을 펼칩니다**

$\chi(t)$ 의 두 성분을 곱해 $\langle S_x\rangle = \frac{\hbar}{2}(\chi^\dagger\sigma_x\chi)$ 를 계산하는 행렬 곱을 보이고, $2\sin\frac{\alpha}{2}\cos\frac{\alpha}{2} = \sin\alpha$ 를 쓰는 마지막 줄을 넣습니다. 최종 식은 기존 식을 유지합니다.

- [ ] **Step 9: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/04-three-dimensions.md | tail -5
git add docs/physics/quantum-mechanics/04-three-dimensions.md docs/physics/quantum-mechanics/images/04-ladder-lz.png
git commit -m "Expand derivations in quantum mechanics chapter 4"
```

---

### Task 5: 5장 동일 입자

**Files:**
- Modify: `docs/physics/quantum-mechanics/05-identical-particles.md` (65–73, 83–87, 255–272, 277, 339–347, 444–467, 498–503줄 부근)
- Create: `docs/physics/quantum-mechanics/images/05-statistics-derivation.png`

- [ ] **Step 1: 교환력 표의 |ψ±|² 전개를 펼칩니다 (65–73줄)**

$|\psi_\pm|^2 = \frac{1}{2}[|\psi_a(x_1)|^2|\psi_b(x_2)|^2 + |\psi_b(x_1)|^2|\psi_a(x_2)|^2 \pm 2\mathrm{Re}(\psi_a^*(x_1)\psi_b(x_1)\psi_b^*(x_2)\psi_a(x_2))]$ 를 보이고, 교차항의 $x_1x_2$ 기댓값이 $\pm 2|\langle x\rangle_{ab}|^2$ 이 되는 두 줄을 넣습니다.

- [ ] **Step 2: 무한 우물 적분 세 개를 펼칩니다 (83–87줄)**

$\langle x^2\rangle_n = a^2(1/3 - 1/(2n^2\pi^2))$ 이 $\int x^2\sin^2(n\pi x/a)dx$ 의 부분 적분에서 나오는 흐름을 세 줄로 보입니다. $\langle x\rangle_{12} = -16a/(9\pi^2)$ 은 3장의 $x_{mn}$ 공식 결과임을 밝힙니다.

- [ ] **Step 3: 페르미 구 적분과 압력을 펼칩니다 (255–272줄)**

$E_{\text{tot}} = \frac{3}{5}NqE_F$ 이 $E_{\text{tot}} = q\int_0^{k_F}\frac{\hbar^2k^2}{2m}\cdot\frac{V}{\pi^2}k^2dk$ 에서 나오는 적분을 보입니다. 이어서 $E_F \propto V^{-2/3}$ 이므로 $\frac{dE_F}{dV} = -\frac{2}{3}\frac{E_F}{V}$ 라는 연쇄 법칙 한 줄을 넣고, $P = \frac{2}{3}\frac{E_{\text{tot}}}{V}$ 를 얻습니다.

- [ ] **Step 4: 델타 결합의 블로흐 조건을 펼칩니다 (339–347줄)**

주기 퍼텐셜에서 각 격자점 $x = na$ 에서 $\psi$ 는 연속이고 $\psi'$ 는 $\frac{2m\alpha}{\hbar^2}\psi(na)$ 만큼 점프한다는 조건을 씁니다. 블로흐 정리 $\psi(x+a) = e^{iKa}\psi(x)$ 를 넣어 두 식을 세우고, 행렬식이 0이라는 조건에서 $\cos Ka = \cos z + \beta\frac{\sin z}{z}$ 를 얻는 흐름을 네 줄로 보입니다. $\beta$ 와 $z$ 의 정의도 함께 밝힙니다.

- [ ] **Step 5: 라그랑주 승수 최대화를 펼칩니다 (444–467줄)**

페르미-디랙을 예로 들어 전 과정을 한 번 보입니다.

1. $\ln W = \sum [d_n\ln d_n - N_n\ln N_n - (d_n-N_n)\ln(d_n-N_n)]$ 로 스털링 근사를 적용합니다.
2. 제약 $F = \ln W + \alpha(N - \sum N_n) + \beta(E - \sum N_n E_n)$ 을 씁니다.
3. $\partial F/\partial N_n = 0$ 에서 $\ln(d_n/N_n - 1) = \alpha + \beta E_n$ 을 얻습니다.
4. $\alpha = -\mu/k_BT$, $\beta = 1/k_BT$ 로 두고 $N_n/d_n$ 을 풉니다.

맥스웰-볼츠만과 보스-아인슈타인은 ���은 절차이며 분모의 $\mp 1$ 만 다르다고 밝힙니다.

- [ ] **Step 6: 통계 분포 유도 흐름도를 그립니다**

`/tmp/qmwork/fig05.py`: 왼쪽에 세 가지 채우는 방법 수, 가운데에 제약 두 개와 라그랑주 승수, 오른쪽에 세 분포 식을 놓는 세 단 상자 그림을 그립니다. 화살표로 흐름을 잇습니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig05.py`

- [ ] **Step 7: 그림을 넣고 번호를 올립니다**

5단계 뒤에 넣고, 기존 Figure 6~7 을 7~8 로 올립니다.

- [ ] **Step 8: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/05-identical-particles.md | tail -5
git add docs/physics/quantum-mechanics/05-identical-particles.md docs/physics/quantum-mechanics/images/05-statistics-derivation.png
git commit -m "Expand derivations in quantum mechanics chapter 5"
```

---

### Task 6: 6장 시간에 무관한 섭동 이론

**Files:**
- Modify: `docs/physics/quantum-mechanics/06-time-independent-perturbation.md` (73–88, 112–118, 291–301, 306–338, 396–402, 496–520줄 부근)
- Create: `docs/physics/quantum-mechanics/images/06-perturbation-orders.png`

- [ ] **Step 1: 1차 파동 함수의 투영과 나눗셈을 펼칩니다 (73–80줄)**

$\lambda^1$ 식에 $\psi_m^0$ 를 내적하면 $E_m^0\langle\psi_m^0|\psi_n^1\rangle + \langle\psi_m^0|H'|\psi_n^0\rangle = E_n^0\langle\psi_m^0|\psi_n^1\rangle$ 이 되는 것을 보이고, $m \ne n$ 이면 $(E_m^0 - E_n^0)\langle\psi_m^0|\psi_n^1\rangle = -\langle\psi_m^0|H'|\psi_n^0\rangle$ 으로 정리해 기존 식을 얻는 흐름을 보입니다.

- [ ] **Step 2: 2차 에너지의 대입과 상쇄를 펼칩니다 (84–88줄)**

$\lambda^2$ 식의 내적에서 $E_n^1$ 항이 $\langle\psi_n^0|\psi_n^1\rangle = 0$ 으로 사라지고, $H^0$ 항이 $E_m^0\langle\psi_m^0|\psi_n^1\rangle$ 이 되어 정리되는 두 줄을 넣습니다. 최종 식은 기존 식을 유지합니다.

- [ ] **Step 3: 비대각 원소의 곱을 합으로 바꿉니다 (112–118줄)**

$\sin\frac{m\pi x}{a}\sin\frac{n\pi x}{a} = \frac{1}{2}[\cos\frac{(m-n)\pi x}{a} - \cos\frac{(m+n)\pi x}{a}]$ 을 명시하고, 적분 $\int_0^{a/2}\cos\frac{k\pi x}{a}dx = \frac{a}{k\pi}\sin\frac{k\pi}{2}$ 를 한 줄로 보입니다. 최종 식은 기존 식을 유지합니다.

- [ ] **Step 4: 상대론 보정의 ⟨p⁴⟩ 를 펼칩니다 (291–301줄)**

$\sqrt{p^2c^2 + m^2c^4} = mc^2\sqrt{1 + p^2/(m^2c^2)} \approx mc^2 + \frac{p^2}{2m} - \frac{p^4}{8m^3c^2}$ 의 테일러 전개를 보입니다. 이어서 $p^2 = 2m(E-V)$ 를 써서 $\langle p^4\rangle = 4m^2\langle(E-V)^2\rangle$ 로 바꾸고, 수소에서 $\langle 1/r\rangle$, $\langle 1/r^2\rangle$ 값을 넣어 기존 결과에 이르는 흐름을 네 줄로 보입니다.

- [ ] **Step 5: 스핀-궤도 결합의 ⟨1/r³⟩ 대입을 펼칩니다 (306–338줄)**

$\mathbf{S}\cdot\mathbf{L} = \frac{1}{2}(J^2 - L^2 - S^2)$ 에서 $j(j+1) - l(l+1) - s(s+1)$ 이 나오는 것을 보이고, $\langle 1/r^3\rangle = 1/(l(l+1/2)(l+1)a^3)$ 을 넣는 한 줄을 넣습니다. 두 보정을 더해 $j$ 만 남는 정리를 두 줄로 보입니다.

- [ ] **Step 6: 섭동 차수 전개 흐름도를 그립니다**

`/tmp/qmwork/fig06.py`: $H = H^0 + \lambda H'$ 에서 출발해 $\lambda^0, \lambda^1, \lambda^2$ 로 갈라지는 세 갈래와, 각 갈래에서 얻는 결과($E^0$, $E^1$, $\psi^1$, $E^2$)를 보이는 흐름도를 그립니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig06.py`

- [ ] **Step 7: 그림을 넣고 번호를 올립니다**

2단계 뒤에 넣고, 기존 Figure 3~5 를 4~6 으로 올립니다.

- [ ] **Step 8: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/06-time-independent-perturbation.md | tail -5
git add docs/physics/quantum-mechanics/06-time-independent-perturbation.md docs/physics/quantum-mechanics/images/06-perturbation-orders.png
git commit -m "Expand derivations in quantum mechanics chapter 6"
```

---

### Task 7: 7장 변분 원리

**Files:**
- Modify: `docs/physics/quantum-mechanics/07-variational-principle.md` (75–83, 107–113, 204–214, 256–267, 338–369, 450–452줄 부근)
- Create: `docs/physics/quantum-mechanics/images/07-variational-minimum.png`

- [ ] **Step 1: 가우스 시험 함수 적분을 펼칩니다 (75–83줄)**

$\int_{-\infty}^{\infty}e^{-2bx^2}dx = \sqrt{\pi/(2b)}$ 와 $\int x^2e^{-2bx^2}dx = \frac{1}{4b}\sqrt{\pi/(2b)}$ 를 보이고, 이 둘이 $\langle T\rangle$ 계산에 들어가는 것을 한 줄로 연결합니다.

- [ ] **Step 2: 3차원 가우스 시험 함수의 적분을 펼칩니다 (204–214줄)**

$\int_0^\infty r e^{-2br^2}dr = 1/(4b)$ 와 구면 적분 부피 인자 $4\pi$ 를 명시하고, $\langle H\rangle(b)$ 가 나오는 두 줄을 넣습니다. $d\langle H\rangle/db = 0$ 에서 최적 $b$ 를 구하는 미분을 한 줄로 보입니다.

- [ ] **Step 3: 헬륨의 전자 반발 적분을 펼칩니다 (256–267줄)**

$\langle 1/r\rangle = Z/a$ 와 $\langle V_{ee}\rangle = \frac{5Z}{8a}\frac{e^2}{4\pi\epsilon_0}$ 의 계산 구조를 설명합니다. 반발 적분이 $Z$ 에 선형으로 비례하는 이유를 한 문장으로 밝히고, 6차원 적분을 다 펼치지는 않습니다. 대신 무엇을 적분하는지(두 전자 좌표, $e^{-2Z(r_1+r_2)/a}/|\mathbf r_1 - \mathbf r_2|$)를 식으로 보입니다.

- [ ] **Step 4: H₂⁺ 의 중첩·직접·교환 적분을 펼칩니다 (338–369줄)**

$I$, $D$, $X$ 의 정의를 적분식으로 쓰고, $x = R/a$ 로 치환해 $I = e^{-x}(1 + x + x^2/3)$ 이 나오는 흐름을 두 줄로 보입니다. $D$ 와 $X$ 는 각각 어떤 적분인지 식으로 보이고 결과를 유지합니다. $\langle H\rangle_\pm$ ��� $E_1[1 + 2(D \pm X)/(1 \pm I)]$ 로 묶이는 한 줄을 넣습니다.

- [ ] **Step 5: 변분 최솟값 그림을 그립니다**

`/tmp/qmwork/fig07.py`: $\langle H\rangle(b)$ 곡선과 $d\langle H\rangle/db = 0$ 인 점, 정확한 바닥 에너지 수평선을 함께 그립니다. 최솟값이 정확한 값보다 위에 있음을 음영으로 표시합니다. 조화 진동자 가우스 시험 함수를 씁니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig07.py`

- [ ] **Step 6: 그림을 넣고 번호를 올립니다**

2단계 뒤에 넣고, 기존 Figure 2~7 을 3~8 로 올립니다.

- [ ] **Step 7: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/07-variational-principle.md | tail -5
git add docs/physics/quantum-mechanics/07-variational-principle.md docs/physics/quantum-mechanics/images/07-variational-minimum.png
git commit -m "Expand derivations in quantum mechanics chapter 7"
```

---

### Task 8: 8장 WKB 근사

**Files:**
- Modify: `docs/physics/quantum-mechanics/08-wkb.md` (58–70, 103–106, 214–224, 313–325, 356–362줄 부근)
- Create: `docs/physics/quantum-mechanics/images/08-wkb-regions.png`

- [ ] **Step 1: WKB 대입의 실수부·허수부 분리를 펼칩니다 (58–70줄)**

$\psi = Ae^{i\phi/\hbar}$ 를 넣고 $\psi'' = (A'' + 2iA'\phi'/\hbar + iA\phi''/\hbar - A\phi'^2/\hbar^2)e^{i\phi/\hbar}$ 를 보입니다. 실수부와 허수부를 나눠 두 방정식을 얻는 과정을 세 줄로 보입니다. 허수부에서 $(A^2\phi')' = 0$ 이 나오는 것을 명시합니다.

- [ ] **Step 2: 가모프 적분의 닫힌 형태를 펼칩니다 (214–219줄)**

$\int_{r_1}^{r_2}\sqrt{\frac{A}{r} - E}\,dr$ 꼴을 $r = r_2\cos^2\theta$ 로 치환하는 방법을 보입니다. 치환하면 $\sqrt{r_2 - r_1}$ 과 $\arccos\sqrt{r_1/r_2}$ 가 나오는 두 줄을 넣고, 최종 식은 기존 식을 유지합니다. 수치는 `scipy.integrate.quad` 로 기존 닫힌 식과 비교합니다.

- [ ] **Step 3: 조화 진동자의 semicircle 적분을 펼칩니다 (356–362줄)**

$x_0^2 = 2E/(m\omega^2)$ 을 대입하고, $\int_{-x_0}^{x_0}\sqrt{x_0^2 - x^2}dx = \frac{\pi x_0^2}{2}$ 가 반원 넓이라는 것을 한 줄로 밝힙니다. 이어서 $\frac{\pi E}{\omega}$ 를 얻습니다.

- [ ] **Step 4: 연결 공식의 계수 맞춤을 펼칩니다 (313–325줄)**

고전적 금지 영역의 지수 해, 고전적 허용 영역의 진동 해, 두 해가 겹치는 구간의 에어리 함수 점근형을 나란히 씁니다. 계수를 맞춰 $\frac{2D}{\sqrt p}\sin(\frac{1}{\hbar}\int p + \frac{\pi}{4})$ 를 얻는 두 줄을 넣습니다.

- [ ] **Step 5: WKB 세 영역 그림을 그립니다**

`/tmp/qmwork/fig08.py`: 퍼텐셜 곡선과 에너지 수평선, 세 영역(금지·허용·금지), 두 전환점 $x_1$, $x_2$, 각 영역의 해 모양을 한 그림에 그립니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig08.py`

- [ ] **Step 6: 그림을 넣고 번호를 올립니다**

4단계 뒤에 넣고, 기존 Figure 5~6 을 6~7 로 올립니다.

- [ ] **Step 7: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/08-wkb.md | tail -5
git add docs/physics/quantum-mechanics/08-wkb.md docs/physics/quantum-mechanics/images/08-wkb-regions.png
git commit -m "Expand derivations in quantum mechanics chapter 8"
```

---

### Task 9: 9장 시간에 의존하는 섭동 이론

**Files:**
- Modify: `docs/physics/quantum-mechanics/09-time-dependent-perturbation.md` (47–54, 84–86, 138–148, 170–183, 285–298, 415줄 부근)
- Create: `docs/physics/quantum-mechanics/images/09-lorentzian-limit.png`

- [ ] **Step 1: 결합 방정식의 투영을 펼칩니다 (47–54줄)**

$\Psi = c_a\psi_ae^{-iE_at/\hbar} + c_b\psi_be^{-iE_bt/\hbar}$ 를 TDSE 에 넣고, $\psi_b^*$ 를 곱해 적분하면 직교성으로 $c_a$ 항이 사라지는 것을 보입니다. 위상 $e^{i\omega_0t}$ 가 남는 두 줄을 넣습니다.

- [ ] **Step 2: 상수 섭동의 |e^{iθ}−1|² 변환을 펼칩니다 (84–86줄)**

$|e^{i\theta} - 1|^2 = (e^{i\theta}-1)(e^{-i\theta}-1) = 2 - 2\cos\theta = 4\sin^2(\theta/2)$ 를 한 줄로 보입니다.

- [ ] **Step 3: 라비 해를 펼칩니다 (170–183줄)**

$\Omega = |V_{ab}|/\hbar$, $\Delta = \omega - \omega_0$, $\Omega' = \sqrt{\Omega^2 + \Delta^2}$ 을 정의하고, 결합 상미분 방정식을 $\ddot c_b + i\Delta\dot c_b + \Omega^2 c_b = 0$ 꼴로 바꾼 뒤 특성 방정식의 근이 $\pm\Omega'/2$ 임을 보입니다. 초기 조건 $c_b(0) = 0$ 에서 $|c_b|^2 = \frac{\Omega^2}{\Omega'^2}\sin^2\frac{\Omega't}{2}$ 를 얻는 흐름을 네 줄로 보입니다.

- [ ] **Step 4: sin²x/x² 적분의 π 값을 펼칩니다 (285–298줄)**

$x = (\omega_0-\omega)t/2$ 로 치환하고 $\int_{-\infty}^{\infty}\frac{\sin^2x}{x^2}dx = \pi$ 를 씁니다. 이 적분이 $\pi$ 인 이유를 $\frac{\sin^2x}{x^2}$ 의 그래프 넓이로 설명하고, $t$ 가 클 때 델타 함수처럼 좁아지는 것을 한 문장으로 밝힙니다.

- [ ] **Step 5: 415줄의 잘못된 수식 표기를 고칩니다**

`$\sqrt{3}/4\pi = Y_0^0 Y_1^0 / \cos\theta$` 는 틀렸습니다. 실제 관계는 $Y_1^0 = \sqrt{3/(4\pi)}\cos\theta$ 입니다. 이 식으로 바로잡고, 각도 적분이 $\frac{1}{\sqrt3}$ 이 되는 근거를 한 줄로 씁니다.

- [ ] **Step 6: 로런츠 극한 그림을 그립니다**

`/tmp/qmwork/fig09.py`: $\frac{\sin^2x}{x^2}$ 를 그리고, $t$ 를 키울수록 좁아지며 넓이가 $\pi$ 로 유지되는 것을 두세 곡선으로 보입니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig09.py`

- [ ] **Step 7: 그림을 넣고 번호를 올립니다**

4단계 뒤에 넣고, 기존 Figure 4~6 을 5~7 로 올립니다.

- [ ] **Step 8: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/09-time-dependent-perturbation.md | tail -5
git add docs/physics/quantum-mechanics/09-time-dependent-perturbation.md docs/physics/quantum-mechanics/images/09-lorentzian-limit.png
git commit -m "Expand derivations in quantum mechanics chapter 9"
```

---

### Task 10: 10장 단열 근사

**Files:**
- Modify: `docs/physics/quantum-mechanics/10-adiabatic.md` (51–55, 136–152, 196–215, 318–343, 382–392줄 부근)
- Create: `docs/physics/quantum-mechanics/images/10-berry-solid-angle.png`

- [ ] **Step 1: 순간 고유 스피너를 풉니다 (196–208줄)**

$\mathbf B(t)$ 가 $z$ 축에서 각 $\alpha$ 만큼 기울어 회전하는 해밀토니안의 2×2 행렬을 쓰고, 고유 벡터 $(\cos\frac\alpha2, e^{i\omega t}\sin\frac\alpha2)$ 를 연립 방정식에서 구하는 두 줄을 넣습니다.

- [ ] **Step 2: 회전장의 정확한 해를 펼칩니다 (210–215줄)**

"결과만 적으면"을 유지하되, 회전 좌표계로 옮기면 해밀토니안이 시간에 무관해진다는 것을 한 문단으로 밝힙니다. 그 뒤 $\lambda = \sqrt{\omega^2 + \omega_1^2 - 2\omega\omega_1\cos\alpha}$ 가 회전 좌표계의 두 준위 차이라는 한 줄을 넣습니다. 전체 유도는 하지 않습니다.

- [ ] **Step 3: 베리 위상 적분을 펼칩니다 (318–343줄)**

$\langle\chi_+|\partial_\phi\chi_+\rangle$ 의 행렬 곱을 보이고, $\sin^2(\alpha/2) = (1-\cos\alpha)/2$ 를 거쳐 $\gamma_+ = -\pi(1-\cos\alpha)$ 를 얻는 두 줄을 넣습니다. $-\pi(1-\cos\alpha)$ 가 구면에서 입체각 $\Omega$ 의 절반임을 밝힙니다.

- [ ] **Step 4: AB 고리의 연산자 제곱을 펼칩니다 (382–392줄)**

$(\frac{d}{d\phi} - i\frac{q\Phi}{2\pi\hbar})^2$ 을 $Ae^{in\phi}$ 에 작용시켜 $(in - i\frac{q\Phi}{2\pi\hbar})^2$ 이 나오는 두 줄을 보입니다. $\Phi_0 = 2\pi\hbar/q$ 를 넣어 기존 에너지 식을 얻습니다.

- [ ] **Step 5: 베리 위상과 입체각 그림을 그립니다**

`/tmp/qmwork/fig10.py`: 구 위의 원뿔(고정 각 $\alpha$)을 그리고, 그 원이 둘러싸는 입체각 $\Omega = 2\pi(1-\cos\alpha)$ 를 음영으로 표시합니다. $\alpha$ 에 따른 $\gamma = -\Omega/2$ 를 오른쪽 패널에 그립니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig10.py`

- [ ] **Step 6: 그림을 넣고 번호를 올립니다**

3단계 뒤에 넣고, 기존 Figure 5~7 을 6~8 로 올립니다.

- [ ] **Step 7: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/10-adiabatic.md | tail -5
git add docs/physics/quantum-mechanics/10-adiabatic.md docs/physics/quantum-mechanics/images/10-berry-solid-angle.png
git commit -m "Expand derivations in quantum mechanics chapter 10"
```

---

### Task 11: 11장 산란

**Files:**
- Modify: `docs/physics/quantum-mechanics/11-scattering.md` (48–53, 68–72, 163–169, 259–265, 269–282줄 부근)
- Create: `docs/physics/quantum-mechanics/images/11-momentum-transfer.png`

- [ ] **Step 1: 단단한 공 미분 단면적의 엉킨 중간 줄을 고칩니다 (48–53줄)**

기존 둘째 줄이 잘못 엮여 있습니다. $b = R\cos(\theta/2)$ 에서 $db/d\theta = -\frac{R}{2}\sin(\theta/2)$ 를 구하고, $D(\theta) = \frac{b}{\sin\theta}\left|\frac{db}{d\theta}\right| = \frac{R\cos(\theta/2)}{\sin\theta}\cdot\frac{R}{2}\sin\frac{\theta}{2}$ 에 $\sin\theta = 2\sin\frac\theta2\cos\frac\theta2$ 를 넣어 $\frac{R^2}{4}$ 를 얻는 깨끗한 세 줄로 바꿉니다.

- [ ] **Step 2: 러더퍼드 미분 단면적을 펼칩니다 (68–72줄)**

$b = \frac{q_1q_2}{8\pi\epsilon_0E}\cot\frac\theta2$ 에서 $db/d\theta = -\frac{q_1q_2}{16\pi\epsilon_0E}\frac{1}{\sin^2(\theta/2)}$ 를 구하는 미분 한 줄과, $D = \frac{b}{\sin\theta}|db/d\theta|$ 에 대입해 기존 식을 얻는 두 줄을 넣습니다.

- [ ] **Step 3: 위상 이동의 경계 맞춤을 펼칩니다 (163–169줄)**

$r = R$ 에서 내부 해와 외부 해 $j_l(kr)\cos\delta_l - n_l(kr)\sin\delta_l$ 이 0이 되는 조건을 씁니다. 두 항을 나누면 $\tan\delta_l = j_l(kR)/n_l(kR)$ 이 나오는 한 줄을 넣습니다.

- [ ] **Step 4: 보른 근사의 그린 함수를 펼칩니다 (259–265줄)**

$(\nabla^2 + k^2)\psi = \frac{2m}{\hbar^2}V\psi$ 에서 출발해, $\frac{e^{ik|\mathbf r - \mathbf r_0|}}{|\mathbf r - \mathbf r_0|}$ 이 $(\nabla^2+k^2)$ 의 그린 함수임을 밝힙니다. 적분 방정식 꼴로 쓰는 두 줄을 넣습니다.

- [ ] **Step 5: 운동량 이동 벡터 그림을 그립니다**

`/tmp/qmwork/fig11.py`: 파수 벡터 $\mathbf k$, $\mathbf k'$, 그 차이 $\boldsymbol\kappa$ 를 그리고 $\kappa = 2k\sin(\theta/2)$ 를 표시합니다. $\theta = 0$ 과 $\pi$ 에서 $\kappa$ 의 값을 함께 적습니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig11.py`

- [ ] **Step 6: 그림을 넣고 번호를 올립니다**

4단계 뒤에 넣고, 기존 Figure 3~6 을 4~7 로 올립니다.

- [ ] **Step 7: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/11-scattering.md | tail -5
git add docs/physics/quantum-mechanics/11-scattering.md docs/physics/quantum-mechanics/images/11-momentum-transfer.png
git commit -m "Expand derivations in quantum mechanics chapter 11"
```

---

### Task 12: 12장 맺음말

**Files:**
- Modify: `docs/physics/quantum-mechanics/12-afterword.md` (100–104, 110–131, 148–165, 389–401줄 부근)
- Create: `docs/physics/quantum-mechanics/images/12-zeno-binomial.png`

- [ ] **Step 1: 단일항 기댓값의 파울리 대수를 펼칩니다 (100–104줄)**

$|\psi\rangle = \frac{1}{\sqrt2}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$ 에 $(\boldsymbol\sigma\cdot\mathbf a)\otimes(\boldsymbol\sigma\cdot\mathbf b)$ 를 작용시켜 네 항을 얻는 계산을 보입니다. $\mathbf a \cdot \mathbf b = \cos\theta$ 로 정리되는 마지막 줄을 넣습니다.

- [ ] **Step 2: 벨 부등식의 덧셈·뺄셈을 펼칩니다 (148–165줄)**

$P(a,b) - P(a,c) = -\int\rho A(a)A(b)[1 - A(b)A(c)]d\lambda$ 로 가는 대수 단계를 보입니다. $A(b)^2 = 1$ 을 끼워 넣는 것이 핵심임을 명시하고, $|A(a)A(b)| \le 1$ 과 $1 - A(b)A(c) \ge 0$ 으로 경계를 얻는 두 줄을 넣습니다.

- [ ] **Step 3: 제논 효과의 이항 근사를 펼칩니다 (389–401줄)**

$[1 - (\frac{\alpha T}{n})^2]^n \approx 1 - \frac{(\alpha T)^2}{n}$ 이 이항 정리 $(1-x)^n \approx 1 - nx$ 에서 나오는 것을 보입니다. $n$ 이 클 때 지수 대신 선형으로 줄어드는 이유를 한 문장으로 밝힙니다.

- [ ] **Step 4: 제논 효과 그림을 그립니다**

`/tmp/qmwork/fig12.py`: 측정 횟수 $n$ 을 가로축으로, 측정 사이 생존 확률을 세로축으로 그립니다. $n$ 이 커질수록 생존 확률이 1에 가까워지는 것을 그리고, 근사식과 정확한 식을 겹쳐 그립니다.

Run: `cd /tmp/qmwork && /tmp/qmfig/bin/python fig12.py`

- [ ] **Step 5: 그림을 넣고 번호를 올립니다**

3단계 뒤에 넣고, 기존 Figure 5 를 6 으로 올립니다.

- [ ] **Step 6: 빌드·린터 확인 후 커밋합니다**

```bash
cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict -q; echo "build=$?"
python3 ~/.claude/skills/korean-writing-skill/scripts/ko-ste-lint.py --mode flavored docs/physics/quantum-mechanics/12-afterword.md | tail -5
git add docs/physics/quantum-mechanics/12-afterword.md docs/physics/quantum-mechanics/images/12-zeno-binomial.png
git commit -m "Expand derivations in quantum mechanics chapter 12"
```

---

### Task 13: 전체 검토

**Files:**
- Modify: `docs/physics/quantum-mechanics/index.md` (읽는 방법 절에 보강 안내 한 문단)
- Verify: 12개 장 전체

- [ ] **Step 1: index 의 읽는 방법에 한 문단을 더합니다**

수식 전개를 어디까지 펼쳤는지와, 비교과서 수준 적분은 결과��� 쓴다는 것을 밝힙니다. 두세 문장으로 씁니다.

- [ ] **Step 2: 전체 빌드를 확인합니다**

Run: `cd /Users/digit82/repo/ai-research && .venv/bin/mkdocs build --strict; echo "build=$?"`
Expected: `build=0`, 경고 0

- [ ] **Step 3: 그림 링크가 모두 살아 있는지 확인합니다**

Run:
```bash
cd /Users/digit82/repo/ai-research/docs/physics/quantum-mechanics && grep -oh 'images/[^)]*\.png' *.md | sort -u > /tmp/qmwork/refs.txt
while read -r p; do [ -f "$p" ] || echo "MISSING $p"; done < /tmp/qmwork/refs.txt; wc -l < /tmp/qmwork/refs.txt
```
Expected: `MISSING` 줄이 없습니다. 참조 파일 수가 76 이상입니다.

- [ ] **Step 4: 쓰이지 않는 그림이 없는지 확인합니다**

Run:
```bash
cd /Users/digit82/repo/ai-research/docs/physics/quantum-mechanics && ls images/*.png | sed 's|^|images/|' | sort > /tmp/qmwork/files.txt; comm -13 /tmp/qmwork/refs.txt /tmp/qmwork/files.txt
```
Expected: 출력이 없습니다. 출력이 있으면 그 그림을 참조하는 장을 확인합니다.

- [ ] **Step 5: Figure 번호가 장 안에서 1부터 연속인지 확인합니다**

Run:
```bash
cd /Users/digit82/repo/ai-research/docs/physics/quantum-mechanics && for f in [01]*.md; do echo "== $f"; grep -o 'Figure [0-9]*' "$f" | tr '\n' ' '; echo; done
```
Expected: 장마다 1부터 빈틈없이 증가합니다.

- [ ] **Step 6: 남은 생략 표현을 찾습니다**

Run: `cd /Users/digit82/repo/ai-research/docs/physics/quantum-mechanics && grep -n '결과만 적으면\|적분을 끝까지 하면\|부분 적분을 두 번' *.md`
Expected: 남은 줄마다 앞뒤에 실제 전개가 있는지 확인합니다. 없으면 그 장의 태스크로 돌아가 펼칩니다.

- [ ] **Step 7: 커밋합니다**

```bash
git add docs/physics/quantum-mechanics/index.md
git commit -m "Note the expanded derivations in the quantum mechanics overview"
```
