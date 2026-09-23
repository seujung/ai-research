# ai-research

AI 트렌드, LLM 서적 정리, 수학 노트를 모아 둔 문서 사이트의 소스입니다.
빌드된 사이트는 <https://seujung.github.io/ai-research/> 에서 볼 수 있습니다.

## 구성

| 경로 | 내용 |
| --- | --- |
| `docs/trends/` | 날짜순으로 쌓이는 트렌드 기록입니다. Material의 blog 플러그인이 목록과 아카이브를 생성합니다. |
| `docs/books/` | 책 한 권이 하나의 디렉토리에 대응하며, 장 순서대로 배열됩니다. |
| `docs/math/` | 주제별 수학 노트입니다. MathJax로 수식을 렌더링합니다. |
| `mkdocs.yml` | 테마, 플러그인, 내비게이션을 포함한 사이트 설정 전체입니다. |
| `overrides/` | 테마 템플릿을 덮어쓸 때 사용하는 디렉토리입니다. |

## 로컬에서 실행하기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

브라우저에서 <http://127.0.0.1:8001> 에 접속하면 확인할 수 있습니다.
파일을 저장하면 자동으로 다시 빌드됩니다.

## 글을 추가하는 방법

### 트렌드

`docs/trends/posts/` 아래에 `YYYY-MM-DD-slug.md` 형식으로 파일을 만들고,
다음과 같이 프론트매터를 작성합니다. `nav` 를 수정할 필요는 없습니다.

```yaml
---
date: 2026-09-23
slug: moe-scaling          # URL에 쓰이는 문자열입니다. 반드시 영문으로 지정합니다.
authors:
  - seujung
categories:
  - 논문
tags:
  - transformer
---
```

`slug` 을 생략하면 제목이 그대로 URL이 됩니다. 제목이 한국어인 경우 주소가
퍼센트 인코딩되어 공유하기 어려워지므로, 항상 영문 `slug` 을 지정하는 편이 좋습니다.

본문에 `<!-- more -->` 를 넣으면 그 앞부분이 목록에 표시되는 요약이 됩니다.

### 서적과 수학

문서를 만든 뒤 `mkdocs.yml` 의 `nav` 에 항목을 추가해야 사이드바에 나타납니다.

## 배포

`dev` 브랜치에 푸시하면 `.github/workflows/deploy.yml` 이 실행되어 자동으로 배포됩니다.
최초 1회에 한해 GitHub 저장소의 **Settings → Pages → Source** 를 **GitHub Actions** 로
변경해 주어야 합니다.
