# =============================================================================
# [파일 설명]
# - 수행 기능: 메일의 관련보도 링크가 열 정적 HTML 상세 페이지를 생성하고 오래된 페이지를 정리합니다.
# - 프로세스: 관련 기사 목록 수집 -> 페이지 URL 부착 -> HTML 렌더링 -> docs/briefings 저장 -> 보존 기간 초과 파일 삭제
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 브리핑 설정, section_results, GitHub Pages 기본 URL, 보존 기간
# - 리턴값/출력: 페이지 생성 여부, 경로, URL, 연결된 뉴스 수, 삭제된 오래된 페이지 수를 담은 dict를 반환합니다.
# =============================================================================

"""
관련보도 상세 페이지 생성 모듈

메일 클라이언트의 접기/펼치기 제한을 피하기 위해, 브리핑 1회당
정적 HTML 상세 페이지 1개를 생성한다.
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytz


DEFAULT_OUTPUT_ROOT = "docs"
DEFAULT_KEEP_DAYS = 7


# [코드 이해 주석]
# - 역할: 입력값을 화면 표시나 후속 처리에 안전한 형태로 변환하는 내부 보조 함수입니다.
# - 호출하는 곳: related_pages._build_related_page_html
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


# [코드 이해 주석]
# - 역할: 입력값을 화면 표시나 후속 처리에 안전한 형태로 변환하는 내부 보조 함수입니다.
# - 호출하는 곳: related_pages._build_related_page_html
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _safe_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "#"
    return html.escape(text, quote=True)


# [코드 이해 주석]
# - 역할: 입력값을 화면 표시나 후속 처리에 안전한 형태로 변환하는 내부 보조 함수입니다.
# - 호출하는 곳: related_pages.generate_related_page
# - 파라미터: value: Any, default: str = 'briefing'
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _safe_slug(value: Any, default: str = "briefing") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9가-힣_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    return text or default


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화하는 내부 보조 함수입니다.
# - 호출하는 곳: related_pages.collect_related_items
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def _normalize_url_for_compare(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.rstrip("/")


# [코드 이해 주석]
# - 역할: GitHub Pages 기본 URL 결정.
# - 호출하는 곳: related_pages.generate_related_page
# - 파라미터: config: Dict[str, Any]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_pages_base_url(config: Dict[str, Any]) -> str:
    """
    GitHub Pages 기본 URL 결정.

    우선순위:
    1. config.pages_base_url
    2. env GITHUB_PAGES_BASE_URL
    3. env GITHUB_REPOSITORY 기반 https://owner.github.io/repo
    """
    configured = str(config.get("pages_base_url") or "").strip()
    if configured:
        return configured.rstrip("/")

    from_env = str(os.getenv("GITHUB_PAGES_BASE_URL") or "").strip()
    if from_env:
        return from_env.rstrip("/")

    repository = str(os.getenv("GITHUB_REPOSITORY") or "").strip()
    if "/" not in repository:
        return ""

    owner, repo = repository.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        return ""

    return f"https://{owner}.github.io/{repo}".rstrip("/")


# [코드 이해 주석]
# - 역할: 여러 입력에서 후속 단계에 필요한 항목을 모읍니다.
# - 호출하는 곳: related_pages._build_related_page_html, related_pages.attach_related_page_urls
# - 파라미터: news: Dict[str, Any]
# - 리턴값: List[Dict[str, str]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 목록을 순회합니다 -> 조건에 맞는 항목을 모읍니다 -> 후속 단계가 사용할 목록/통계를 반환합니다.
def collect_related_items(news: Dict[str, Any]) -> List[Dict[str, str]]:
    titles = news.get("group_article_titles") or []
    urls = news.get("group_article_urls") or []
    sources = news.get("group_article_sources") or []
    if not isinstance(sources, list):
        sources = []
    main_url = _normalize_url_for_compare(news.get("url"))

    related_items: List[Dict[str, str]] = []
    seen_urls = set()
    seen_titles = set()

    for index, (title, url) in enumerate(zip(titles, urls)):
        title_text = str(title or "").strip()
        url_text = str(url or "").strip()
        source_text = str(sources[index] or "").strip() if index < len(sources) else ""
        normalized_url = _normalize_url_for_compare(url_text)
        normalized_title = re.sub(r"\s+", " ", title_text).strip().lower()

        if not title_text or not url_text or url_text == "#":
            continue
        if normalized_url and normalized_url in seen_urls:
            continue
        if normalized_title and normalized_title in seen_titles:
            continue

        if not source_text and main_url and normalized_url == main_url:
            source_text = str(news.get("source") or "").strip()

        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)
        related_items.append({
            "title": title_text,
            "url": url_text,
            "source": source_text,
            "is_main": "true" if main_url and normalized_url == main_url else "",
        })

    return related_items


# [코드 이해 주석]
# - 역할: 생성된 URL이나 메타데이터를 뉴스 dict에 연결합니다.
# - 호출하는 곳: related_pages.generate_related_page
# - 파라미터: section_results: List[Dict[str, Any]], page_url: str
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def attach_related_page_urls(
    section_results: List[Dict[str, Any]],
    page_url: str,
) -> int:
    linked_count = 0
    for section_index, section_result in enumerate(section_results or [], 1):
        for news_index, news in enumerate(section_result.get("summaries") or [], 1):
            related_items = collect_related_items(news)
            if not related_items:
                continue
            anchor = f"news-{section_index}-{news_index}"
            news["related_reports_count"] = len(related_items)
            news["related_reports_url"] = f"{page_url}#{anchor}"
            linked_count += 1
    return linked_count


# [코드 이해 주석]
# - 역할: 보존 기간을 지난 생성 파일을 삭제하고 정리 결과를 반환합니다.
# - 호출하는 곳: related_pages.generate_related_page
# - 파라미터: directory: str, keep_days: int = DEFAULT_KEEP_DAYS
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def cleanup_old_pages(directory: str, keep_days: int = DEFAULT_KEEP_DAYS) -> int:
    if not os.path.isdir(directory):
        return 0

    kst = pytz.timezone("Asia/Seoul")
    cutoff = datetime.now(kst) - timedelta(days=max(int(keep_days or 0), 1))
    removed_count = 0

    for name in os.listdir(directory):
        if not name.endswith(".html"):
            continue
        path = os.path.join(directory, name)
        try:
            page_time = datetime.strptime(name.replace(".html", ""), "%Y-%m-%d-%H%M%S")
            modified = kst.localize(page_time) if hasattr(kst, "localize") else page_time.replace(tzinfo=kst)
        except Exception:
            try:
                modified = datetime.fromtimestamp(os.path.getmtime(path), kst)
            except Exception:
                continue
        if modified >= cutoff:
            continue
        try:
            os.remove(path)
            removed_count += 1
        except Exception:
            continue

    return removed_count


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 내부에서 사용할 출력 구조를 만드는 보조 함수입니다.
# - 호출하는 곳: related_pages.generate_related_page
# - 파라미터: briefing_name: str, subject_prefix: str, section_results: List[Dict[str, Any]], generated_at_text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 내부용 문자열/dict 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def _build_related_page_html(
    briefing_name: str,
    subject_prefix: str,
    section_results: List[Dict[str, Any]],
    generated_at_text: str,
) -> str:
    section_html = ""

    for section_index, section_result in enumerate(section_results or [], 1):
        section_title = section_result.get("section_name") or f"뉴스 섹션 {section_index}"
        news_html = ""

        for news_index, news in enumerate(section_result.get("summaries") or [], 1):
            related_items = collect_related_items(news)
            if not related_items:
                continue

            anchor = f"news-{section_index}-{news_index}"
            related_rows = ""
            for item_index, item in enumerate(related_items, 1):
                main_badge = ""
                if item.get("is_main"):
                    main_badge = '<span class="badge">대표</span>'
                source_label = ""
                if item.get("source"):
                    source_label = f'<span class="source">{_safe_text(item.get("source"))}</span>'
                related_rows += f"""
                    <li>
                        {source_label}
                        <a href="{_safe_url(item.get("url"))}" target="_blank" rel="noopener noreferrer">
                            {_safe_text(item.get("title"))}
                        </a>
                        {main_badge}
                    </li>
                """

            news_html += f"""
                <article id="{anchor}" class="news-card">
                    <h2>
                        <a href="{_safe_url(news.get("url"))}" target="_blank" rel="noopener noreferrer">
                            {_safe_text(news.get("title") or "제목 없음")}
                        </a>
                    </h2>
                    <p>{_safe_text(news.get("summary"))}</p>
                    <div class="related-count">관련보도 {len(related_items)}건</div>
                    <ol>
                        {related_rows}
                    </ol>
                </article>
            """

        if news_html:
            section_html += f"""
                <section>
                    <h1>{_safe_text(section_title)}</h1>
                    {news_html}
                </section>
            """

    if not section_html:
        section_html = '<p class="empty">관련보도 상세 목록이 없습니다.</p>'

    title = f"{generated_at_text} - {subject_prefix or briefing_name}"
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="noindex,nofollow">
    <title>{_safe_text(title)}</title>
    <style>
        body {{
            margin: 0;
            padding: 24px;
            background: #f4f4f5;
            color: #18181b;
            font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
        }}
        main {{
            max-width: 920px;
            margin: 0 auto;
        }}
        header {{
            margin: 0 0 22px 0;
            padding: 0 0 16px 0;
            border-bottom: 2px solid #18181b;
        }}
        header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            line-height: 1.35;
        }}
        header p {{
            margin: 0;
            color: #52525b;
            font-size: 13px;
            line-height: 1.6;
        }}
        section {{
            margin: 0 0 28px 0;
        }}
        section > h1 {{
            margin: 0 0 12px 0;
            font-size: 18px;
            line-height: 1.35;
            color: #1d4ed8;
        }}
        .news-card {{
            margin: 0 0 14px 0;
            padding: 14px 16px;
            background: #ffffff;
            border: 1px solid #d4d4d8;
            border-radius: 6px;
        }}
        body.related-filtered section:not(.selected-section),
        body.related-filtered .news-card:not(.selected-related) {{
            display: none;
        }}
        .related-count {{
            margin: 10px 0 6px 0;
            color: #71717a;
            font-size: 12px;
            font-weight: 700;
        }}
        h2 {{
            margin: 0 0 8px 0;
            font-size: 16px;
            line-height: 1.45;
        }}
        p {{
            margin: 0 0 10px 0;
            font-size: 13px;
            line-height: 1.65;
        }}
        ol {{
            margin: 0;
            padding-left: 22px;
        }}
        li {{
            margin: 0 0 7px 0;
            font-size: 13px;
            line-height: 1.55;
        }}
        a {{
            color: #1d4ed8;
            font-weight: 700;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .badge {{
            display: inline-block;
            margin-left: 6px;
            color: #71717a;
            font-size: 11px;
            font-weight: 800;
        }}
        .source {{
            display: inline-block;
            margin-right: 7px;
            color: #52525b;
            font-size: 12px;
            font-weight: 800;
        }}
        .empty {{
            padding: 14px 16px;
            background: #ffffff;
            border: 1px solid #d4d4d8;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <main>
        <header>
            <h1>{_safe_text(title)}</h1>
            <p>메일에서 선택한 뉴스의 관련보도 목록입니다. 이 페이지는 자동 생성되며 일정 기간 후 삭제됩니다.</p>
        </header>
        {section_html}
    </main>
    <script>
        (function () {{
            function getHashId() {{
                var raw = window.location.hash ? window.location.hash.slice(1) : "";
                try {{
                    return decodeURIComponent(raw);
                }} catch (error) {{
                    return raw;
                }}
            }}

            function filterSelectedNews() {{
                var id = getHashId();
                var cards = document.querySelectorAll(".news-card");
                var sections = document.querySelectorAll("section");

                document.body.classList.remove("related-filtered");
                cards.forEach(function (card) {{
                    card.classList.remove("selected-related");
                }});
                sections.forEach(function (section) {{
                    section.classList.remove("selected-section");
                }});

                if (!id) {{
                    return;
                }}

                var selected = document.getElementById(id);
                if (!selected || !selected.classList.contains("news-card")) {{
                    return;
                }}

                document.body.classList.add("related-filtered");
                selected.classList.add("selected-related");

                var section = selected.closest("section");
                if (section) {{
                    section.classList.add("selected-section");
                }}

                window.scrollTo(0, 0);
            }}

            window.addEventListener("DOMContentLoaded", filterSelectedNews);
            window.addEventListener("hashchange", filterSelectedNews);
        }})();
    </script>
</body>
</html>
"""


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: main.main
# - 파라미터: config: Dict[str, Any], config_slug: str, briefing_name: str, subject_prefix: str, section_results:
# List[Dict[str, Any]], output_root: str = DEFAULT_OUTPUT_ROOT, keep_days: int = DEFAULT_KEEP_DAYS
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def generate_related_page(
    *,
    config: Dict[str, Any],
    config_slug: str,
    briefing_name: str,
    subject_prefix: str,
    section_results: List[Dict[str, Any]],
    output_root: str = DEFAULT_OUTPUT_ROOT,
    keep_days: int = DEFAULT_KEEP_DAYS,
) -> Dict[str, Any]:
    pages_base_url = get_pages_base_url(config)
    if not pages_base_url:
        return {
            "generated": False,
            "reason": "pages_base_url 없음",
            "linked_count": 0,
            "removed_count": 0,
        }

    kst = pytz.timezone("Asia/Seoul")
    now = datetime.now(kst)
    generated_at_text = now.strftime("%Y년 %m월 %d일")
    filename = now.strftime("%Y-%m-%d-%H%M%S.html")

    safe_config_slug = _safe_slug(config_slug)
    relative_dir = os.path.join("briefings", safe_config_slug)
    output_dir = os.path.join(output_root, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    removed_count = cleanup_old_pages(output_dir, keep_days=keep_days)

    relative_url = f"briefings/{safe_config_slug}/{filename}"
    page_url = f"{pages_base_url}/{relative_url}"
    linked_count = attach_related_page_urls(section_results, page_url)

    if linked_count <= 0:
        return {
            "generated": False,
            "reason": "관련보도 없음",
            "linked_count": 0,
            "removed_count": removed_count,
        }

    output_path = os.path.join(output_dir, filename)
    html_content = _build_related_page_html(
        briefing_name=briefing_name,
        subject_prefix=subject_prefix,
        section_results=section_results,
        generated_at_text=generated_at_text,
    )

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html_content)

    return {
        "generated": True,
        "path": output_path,
        "url": page_url,
        "linked_count": linked_count,
        "removed_count": removed_count,
    }
