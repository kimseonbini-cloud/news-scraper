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


def _safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


def _safe_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "#"
    return html.escape(text, quote=True)


def _safe_slug(value: Any, default: str = "briefing") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9가-힣_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    return text or default


def _normalize_url_for_compare(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.rstrip("/")


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


def collect_related_items(news: Dict[str, Any]) -> List[Dict[str, str]]:
    titles = news.get("group_article_titles") or []
    urls = news.get("group_article_urls") or []
    main_url = _normalize_url_for_compare(news.get("url"))

    related_items: List[Dict[str, str]] = []
    seen_urls = set()
    seen_titles = set()

    for title, url in zip(titles, urls):
        title_text = str(title or "").strip()
        url_text = str(url or "").strip()
        normalized_url = _normalize_url_for_compare(url_text)
        normalized_title = re.sub(r"\s+", " ", title_text).strip().lower()

        if not title_text or not url_text or url_text == "#":
            continue
        if normalized_url and normalized_url in seen_urls:
            continue
        if normalized_title and normalized_title in seen_titles:
            continue

        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)
        related_items.append({
            "title": title_text,
            "url": url_text,
            "is_main": "true" if main_url and normalized_url == main_url else "",
        })

    return related_items


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
                related_rows += f"""
                    <li>
                        <a href="{_safe_url(item.get("url"))}" target="_blank" rel="noopener noreferrer">
                            {_safe_text(item.get("title"))}
                        </a>
                        {main_badge}
                    </li>
                """

            news_html += f"""
                <article id="{anchor}" class="news-card">
                    <div class="meta">{_safe_text(section_title)} · 관련보도 {len(related_items)}건</div>
                    <h2>
                        <a href="{_safe_url(news.get("url"))}" target="_blank" rel="noopener noreferrer">
                            {_safe_text(news.get("title") or "제목 없음")}
                        </a>
                    </h2>
                    <p>{_safe_text(news.get("summary"))}</p>
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
        .news-card:target {{
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16);
        }}
        .meta {{
            margin: 0 0 6px 0;
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
            <p>메일의 관련보도 전체 목록입니다. 이 페이지는 자동 생성되며 일정 기간 후 삭제됩니다.</p>
        </header>
        {section_html}
    </main>
</body>
</html>
"""


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
    generated_at_text = now.strftime("%Y년 %m월 %d일 %H:%M")
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
