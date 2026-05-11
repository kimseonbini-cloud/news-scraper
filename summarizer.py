"""
OpenAI API를 사용한 뉴스 요약 모듈
"""
import os
import logging
import time
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 모델 설정
MODEL = "gpt-4o-mini"


def _safe_text(value: Any) -> str:
    """
    None 방지용 문자열 변환
    """
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 3, min_value: int = 1, max_value: int = 5) -> int:
    """
    중요도 점수 안전 변환
    """
    try:
        number = int(value)
    except Exception:
        number = default

    if number < min_value:
        return min_value

    if number > max_value:
        return max_value

    return number


def summarize_article(article: Dict, max_length: int = 220) -> Dict:
    """
    단일 기사 요약

    Args:
        article: {
            'title': str,
            'content': str,
            'description': str,
            'url': str,
            'keyword': str,
            'published_at': str,
            'importance_score': int,
            'category': str,
            'source': str
        }
        max_length: 요약 최대 길이, 기본 220자

    Returns:
        {
            'title': str,
            'summary': str,
            'url': str,
            'keyword': str,
            'published_at': str,
            'importance_score': int,
            'category': str,
            'source': str,
            'tokens_used': int
        }
    """
    title = _safe_text(article.get("title", "제목 없음"))
    content = _safe_text(article.get("content") or article.get("description"))
    url = _safe_text(article.get("url", "#"))
    keyword = _safe_text(article.get("keyword", ""))
    published_at = _safe_text(article.get("published_at", ""))
    importance_score = _safe_int(article.get("importance_score", 3))
    category = _safe_text(article.get("category")) or "기타"

    # 언론사명 유지
    source = (
        _safe_text(article.get("source"))
        or _safe_text(article.get("press"))
        or _safe_text(article.get("publisher"))
        or _safe_text(article.get("media"))
        or "언론사 미상"
    )

    try:
        prompt = f"""
다음 뉴스 기사를 {max_length}자 이내로 요약해주세요.

[요약 기준]
1. 핵심 내용만 간결하게 정리하세요.
2. 기사 제목과 내용에 없는 사실은 추가하지 마세요.
3. 기업명, 서비스명, 수치, 일정은 원문에 있는 경우에만 포함하세요.
4. 추측성 표현은 쓰지 마세요.
5. 문장은 자연스러운 한국어로 작성하세요.
6. 요약은 1~2문장으로 작성하세요.

제목: {title}
내용: {content}

요약:
"""

        logger.info(f"📝 요약 중: {title[:30]}...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 기사를 간결하고 정확하게 요약합니다. "
                        "원문에 없는 사실을 추가하지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=300
        )

        summary = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info(f"✅ 요약 완료: {len(summary)}자 (토큰: {tokens_used})")

        return {
            "title": title,
            "summary": summary,
            "url": url,
            "keyword": keyword,
            "published_at": published_at,
            "importance_score": importance_score,
            "category": category,
            "source": source,
            "tokens_used": tokens_used
        }

    except Exception as e:
        logger.error(f"❌ 요약 실패: {e}")

        fallback_summary = content[:max_length] + "..." if len(content) > max_length else content

        return {
            "title": title,
            "summary": fallback_summary,
            "url": url,
            "keyword": keyword,
            "published_at": published_at,
            "importance_score": importance_score,
            "category": category,
            "source": source,
            "tokens_used": 0,
            "error": str(e)
        }


def summarize_batch(articles: List[Dict], delay: float = 1.0) -> List[Dict]:
    """
    여러 기사 일괄 요약

    Args:
        articles: 기사 리스트
        delay: API 호출 간 대기 시간, 초

    Returns:
        요약 결과 리스트
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🤖 {len(articles)}개 기사 요약 시작")
    logger.info(f"{'=' * 60}\n")

    summaries = []
    total_tokens = 0

    for i, article in enumerate(articles, 1):
        logger.info(f"진행: {i}/{len(articles)}")

        summary = summarize_article(article)
        summaries.append(summary)

        total_tokens += summary.get("tokens_used", 0)

        if i < len(articles):
            time.sleep(delay)

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ 요약 완료")
    logger.info(f"총 토큰: {total_tokens:,}")
    logger.info(f"예상 비용: ${total_tokens * 0.00015:.4f} USD")
    logger.info(f"{'=' * 60}\n")

    return summaries