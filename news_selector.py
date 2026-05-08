"""
OpenAI API를 사용한 뉴스 선별 모듈

역할:
- 네이버 뉴스 API로 수집된 전체 뉴스 후보 중
- 주제 적합성, 중요도, 중복 여부를 기준으로
- 요약할 뉴스 10개를 먼저 선택한다.
- 선택된 뉴스에 중요도 점수, 카테고리를 함께 부여한다.
"""

import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def _build_candidate_text(news_list: List[Dict]) -> str:
    """
    OpenAI에 전달할 뉴스 후보 목록 텍스트 생성
    """
    lines = []

    for idx, news in enumerate(news_list, 1):
        title = _safe_text(news.get("title"))
        description = _safe_text(news.get("description"))
        keyword = _safe_text(news.get("keyword"))
        date = _safe_text(news.get("date"))
        published_at = _safe_text(news.get("published_at"))
        url = _safe_text(news.get("url"))

        lines.append(
            f"""
[{idx}]
키워드: {keyword}
제목: {title}
설명: {description}
날짜: {published_at or date}
URL: {url}
""".strip()
        )

    return "\n\n".join(lines)


def _extract_json(content: str) -> Dict:
    """
    OpenAI 응답에서 JSON 파싱.
    원칙적으로 JSON만 오게 하지만, 혹시 코드블록이 섞이는 경우를 대비한다.
    """
    content = _safe_text(content)

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)


def _fallback_select(news_list: List[Dict], limit: int) -> List[Dict]:
    """
    OpenAI 선별 실패 시 안전 fallback.
    앞에서 limit개를 사용하되, 요약 단계에 필요한 필드를 기본값으로 넣는다.
    """
    fallback_news = news_list[:limit]

    for news in fallback_news:
        news["content"] = news.get("description", "")
        news["importance_score"] = _safe_int(news.get("importance_score", 3))
        news["category"] = _safe_text(news.get("category")) or "기타"

    return fallback_news


def select_important_news(
    news_list: List[Dict],
    topic_name: str,
    topic_description: str,
    limit: int = 10
) -> List[Dict]:
    """
    전체 뉴스 후보 중 OpenAI가 중요하고 주제에 맞는 뉴스만 선택

    Args:
        news_list: 네이버 API로 수집된 전체 뉴스 리스트
        topic_name: 섹션명 예: "롯데그룹 의료뉴스브리핑"
        topic_description: 어떤 뉴스를 뽑아야 하는지 설명
        limit: 최종 선택 개수

    Returns:
        선택된 뉴스 리스트.
        각 뉴스에는 아래 필드가 추가된다.
        - importance_score: 중요도 점수, 1~5
        - category: 자동 분류 카테고리
        - content: summarizer.py에서 사용할 요약 대상 본문
    """
    if not news_list:
        logger.warning("선택할 뉴스 후보가 없습니다.")
        return []

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🧠 OpenAI 뉴스 선별 시작: {topic_name}")
    logger.info(f"후보 뉴스 수: {len(news_list)}개")
    logger.info(f"최종 선택 목표: {limit}개")
    logger.info(f"{'=' * 60}")

    candidate_text = _build_candidate_text(news_list)

    prompt = f"""
아래는 네이버 뉴스 API로 수집한 뉴스 후보 목록입니다.

당신의 역할은 뉴스 편집자입니다.
아래 기준으로 최종 뉴스 {limit}개 이하를 선택하세요.

[브리핑 이름]
{topic_name}

[선택해야 하는 뉴스 주제]
{topic_description}

[선택 기준]
1. 주제와 직접 관련 있는 뉴스만 선택
2. 단순 홍보성, 연관성이 약한 기사, 키워드만 걸린 기사는 제외
3. 같은 내용을 다룬 중복된 기사는 1개만 선택
4. 기업 의사결정, 사업 전략, 실적, 투자, 제휴, 인사, 규제, 기술 변화처럼 중요도가 높은 기사 우선
5. 제목만 자극적인 기사보다 실제 내용이 분명한 기사 우선
6. 가능하면 최신 기사 우선
7. 최종적으로 정확히 {limit}개 이하만 선택
8. 선택할 뉴스가 부족하면 억지로 {limit}개를 채우지 말고, 주제에 맞는 것만 선택

[중요도 점수 기준]
5점: 기업 전략, 대형 투자, 실적, 규제, 산업 변화에 직접 영향
4점: 사업 방향, 기술 도입, 제휴, 주요 서비스 변화와 관련
3점: 참고할 만한 일반 산업 뉴스
2점: 관련성은 있으나 영향도가 낮은 뉴스
1점: 키워드는 있으나 중요도가 낮은 뉴스

[카테고리 기준]
브리핑 주제에 맞춰 짧게 분류하세요.

의료 뉴스 카테고리 예시:
EMR, 병원IT, 의료AI, 디지털헬스케어, 정책·규제, 의료데이터, 클라우드, 보안, 기타

롯데 뉴스 카테고리 예시:
롯데이노베이트, 그룹전략, 실적, 투자·제휴, 신사업, 인사, 리스크, 유통, 기타

기타 브리핑도 주제에 맞게 분류하세요.

[출력 형식]
반드시 JSON만 출력하세요.
설명 문장, 마크다운, 코드블록은 쓰지 마세요.

형식:
{{
  "selected": [
    {{
      "index": 1,
      "importance_score": 5,
      "category": "사업전략"
    }},
    {{
      "index": 5,
      "importance_score": 4,
      "category": "의료AI"
    }}
  ]
}}

[뉴스 후보 목록]
{candidate_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 뉴스 후보 중 중요하고 주제에 맞는 기사만 선별하는 편집자입니다. "
                        "반드시 JSON만 출력합니다. "
                        "선택한 각 기사에는 index, importance_score, category만 포함합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200
        )

        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info(f"🧾 뉴스 선별 토큰 사용량: {tokens_used}")

        try:
            result = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("❌ OpenAI 응답 JSON 파싱 실패")
            logger.error(content)
            return _fallback_select(news_list, limit)

        selected_items = result.get("selected", [])

        if not selected_items:
            logger.warning("⚠️ OpenAI가 선택한 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(news_list, limit)

        selected_news = []
        seen_urls = set()
        seen_titles = set()

        for item in selected_items:
            try:
                selected_index = item.get("index")
                idx = int(selected_index) - 1
            except Exception:
                continue

            if idx < 0 or idx >= len(news_list):
                continue

            news = news_list[idx]

            url = _safe_text(news.get("url"))
            title = _safe_text(news.get("title"))

            # URL 또는 제목 기준 최소 중복 방지
            if url and url in seen_urls:
                continue

            if title and title in seen_titles:
                continue

            if url:
                seen_urls.add(url)

            if title:
                seen_titles.add(title)

            news["importance_score"] = _safe_int(item.get("importance_score", 3))
            news["category"] = _safe_text(item.get("category")) or "기타"
            news["content"] = news.get("description", "")

            selected_news.append(news)

            if len(selected_news) >= limit:
                break

        if not selected_news:
            logger.warning("⚠️ 유효하게 선별된 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(news_list, limit)

        logger.info(f"✅ OpenAI 뉴스 선별 완료: {len(selected_news)}개 선택")

        for i, news in enumerate(selected_news, 1):
            logger.info(
                f"   [{i}] "
                f"중요도 {news.get('importance_score', 3)} | "
                f"{news.get('category', '기타')} | "
                f"{news.get('title', '')[:60]}"
            )

        return selected_news

    except Exception as e:
        logger.error(f"❌ OpenAI 뉴스 선별 실패: {e}")
        logger.warning(f"⚠️ 실패 시 후보 뉴스 앞에서 {limit}개 사용")
        return _fallback_select(news_list, limit)