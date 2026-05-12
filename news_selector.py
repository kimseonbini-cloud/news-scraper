"""
OpenAI API를 사용한 뉴스 선별 모듈

역할:
- 네이버 뉴스 API로 수집된 전체 뉴스 후보 중
- 주제 적합성, 중요도, 중복 여부를 기준으로
- 요약할 뉴스 후보를 먼저 선택한다.
- 선택된 뉴스에 중요도 점수, 카테고리를 함께 부여한다.
- 선택된 뉴스 중 같은 사건을 다룬 중복 기사들은 LLM으로 그룹화하여 1개만 남긴다.
"""

import os
import json
import logging
from typing import List, Dict, Any
from urllib.parse import urlparse

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 뉴스 선별 모델
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


def _normalize_url(url: Any) -> str:
    """
    URL 완전 중복 비교용 정규화

    처리:
    - scheme 제거
    - www. 제거
    - query string 제외
    - fragment 제외
    - 마지막 slash 제거
    """
    url = _safe_text(url)

    if not url:
        return ""

    try:
        parsed = urlparse(url)

        domain = parsed.netloc.lower().strip()
        path = parsed.path.strip()

        if domain.startswith("www."):
            domain = domain[4:]

        normalized = f"{domain}{path}".rstrip("/")
        return normalized

    except Exception:
        return url.lower().strip().rstrip("/")


def _deduplicate_by_url(news_list: List[Dict], log_prefix: str = "후보") -> List[Dict]:
    """
    URL이 완전히 같은 기사만 제거한다.

    같은 사건 여부는 LLM이 따로 판단하고,
    여기서는 완전히 같은 링크만 1차 제거한다.
    """
    deduped_news = []
    seen_urls = set()

    for news in news_list:
        url_candidates = [
            _normalize_url(news.get("url")),
            _normalize_url(news.get("link")),
            _normalize_url(news.get("originallink")),
        ]

        url_candidates = [url for url in url_candidates if url]

        duplicated = False

        for url in url_candidates:
            if url in seen_urls:
                logger.info(
                    f"⏭️ {log_prefix} URL 중복 제외: "
                    f"{_safe_text(news.get('title'))[:70]}"
                )
                duplicated = True
                break

        if duplicated:
            continue

        for url in url_candidates:
            seen_urls.add(url)

        deduped_news.append(news)

    logger.info(
        f"🧹 {log_prefix} URL 중복 제거 완료: {len(news_list)}개 → {len(deduped_news)}개"
    )

    return deduped_news


def _build_candidate_text(news_list: List[Dict]) -> str:
    """
    OpenAI에 전달할 뉴스 후보 목록 텍스트 생성
    """
    lines = []

    for idx, news in enumerate(news_list, 1):
        title = _safe_text(news.get("title"))
        description = _safe_text(news.get("description"))
        keyword = _safe_text(news.get("keyword"))
        source = _safe_text(news.get("source"))
        date = _safe_text(news.get("date"))
        published_at = _safe_text(news.get("published_at"))
        url = _safe_text(news.get("url"))

        lines.append(
            f"""
[{idx}]
키워드: {keyword}
언론사: {source}
제목: {title}
설명: {description}
날짜: {published_at or date}
URL: {url}
""".strip()
        )

    return "\n\n".join(lines)


def _build_event_dedup_text(news_list: List[Dict]) -> str:
    """
    LLM 사건 중복 제거용 뉴스 목록 텍스트 생성
    """
    lines = []

    for idx, news in enumerate(news_list, 1):
        source = _safe_text(news.get("source"))
        category = _safe_text(news.get("category"))
        importance_score = _safe_text(news.get("importance_score"))
        title = _safe_text(news.get("title"))
        description = _safe_text(
            news.get("description")
            or news.get("summary")
            or news.get("content")
        )
        published_at = _safe_text(news.get("published_at"))
        url = _safe_text(news.get("url"))

        lines.append(
            f"""
[{idx}]
언론사: {source}
카테고리: {category}
중요도: {importance_score}
제목: {title}
설명: {description}
발행일: {published_at}
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


def _prepare_selected_news(
    news: Dict,
    importance_score: Any = 3,
    category: Any = "기타"
) -> Dict:
    """
    요약 단계로 넘기기 전에 필요한 필드 보강
    """
    news["importance_score"] = _safe_int(importance_score)
    news["category"] = _safe_text(category) or "기타"
    news["content"] = news.get("description", "")

    return news


def _fallback_select(news_list: List[Dict], limit: int) -> List[Dict]:
    """
    OpenAI 선별 실패 시 안전 fallback.
    URL 중복 제거 후 앞에서 limit개를 사용하되,
    요약 단계에 필요한 필드를 기본값으로 넣는다.
    """
    deduped_news = _deduplicate_by_url(news_list, log_prefix="fallback")
    fallback_news = deduped_news[:limit]

    for news in fallback_news:
        _prepare_selected_news(
            news=news,
            importance_score=news.get("importance_score", 3),
            category=news.get("category") or "기타"
        )

    return fallback_news


def _deduplicate_by_llm_event_group(
    news_list: List[Dict],
    limit: int
) -> List[Dict]:
    """
    LLM을 사용해 모든 뉴스를 사건 단위로 그룹화하고,
    각 사건 그룹에서 대표 기사 1개만 남긴다.

    핵심:
    - 중복 그룹만 선택적으로 찾게 하지 않는다.
    - 모든 기사에 반드시 event_group을 배정하게 한다.
    - 같은 사건이면 같은 그룹에 넣는다.
    - 각 그룹에서 representative_index 1개만 최종 유지한다.
    """
    if not news_list:
        return []

    if len(news_list) <= 1:
        return news_list[:limit]

    news_text = _build_event_dedup_text(news_list)

    prompt = f"""
아래 뉴스 목록을 '사건 단위'로 그룹화하세요.

[목표]
뉴스 제목이나 언론사가 달라도 실제로 같은 사건을 다룬 기사라면 같은 event_group으로 묶고,
각 event_group에서 대표 기사 1개만 남기세요.

[반드시 지켜야 할 기준]
1. 모든 뉴스 index는 반드시 정확히 하나의 event_group에 포함되어야 합니다.
2. 같은 회사, 병원, 기관, 학회, 정부기관, 지자체가 같은 발표·공개·론칭·계약·제휴·실적·공시·행사·정책·서비스를 다룬 기사는 같은 사건입니다.
3. 같은 솔루션, 같은 플랫폼, 같은 서비스, 같은 기술, 같은 시스템 공개를 다룬 기사는 제목이 달라도 같은 사건입니다.
4. 여러 언론사가 같은 보도자료를 받아쓴 기사는 같은 사건입니다.
5. 제목 표현, 언론사, 문장 구조, 중요도, 카테고리가 달라도 실제 사건이 같으면 같은 event_group입니다.
6. 단순히 같은 회사나 같은 질병 분야가 언급됐다는 이유만으로는 같은 사건이 아닙니다.
7. 서로 다른 발표, 다른 서비스, 다른 날짜의 별도 사건이면 다른 event_group입니다.
8. 각 event_group의 representative_index는 가장 정보가 구체적이고 요약하기 좋은 기사 1개로 고르세요.
9. 최종적으로 같은 event_group에서는 representative_index만 남기고 나머지는 제거됩니다.

[예시]
- "의료영상 특화 LLM 플랫폼 공개"
- "의료영상 특화 LLM 플랫폼 론칭"
- "LLM 의료영상 플랫폼 공개"
위처럼 같은 회사의 같은 플랫폼 공개를 다룬 기사들은 같은 event_group입니다.

- "온열질환 예측정보 서비스 제공"
- "폭염 대비 온열질환 발생 예측정보 서비스"
- "온열질환 위험 사전 예측"
위처럼 같은 예측정보 서비스 발표를 다룬 기사들은 같은 event_group입니다.

[출력 규칙]
반드시 JSON만 출력하세요.
설명 문장, 마크다운, 코드블록은 쓰지 마세요.

[출력 형식]
{{
  "event_groups": [
    {{
      "event_group": "제이엘케이 의료영상 특화 LLM 플랫폼 공개",
      "representative_index": 2,
      "indexes": [2, 4, 6]
    }},
    {{
      "event_group": "온열질환 예측정보 서비스 제공",
      "representative_index": 8,
      "indexes": [8, 9, 10]
    }},
    {{
      "event_group": "다른 독립 뉴스",
      "representative_index": 1,
      "indexes": [1]
    }}
  ]
}}

[뉴스 목록]
{news_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 뉴스 편집 데스크입니다. "
                        "모든 뉴스를 사건 단위로 그룹화하고, "
                        "각 사건 그룹에서 대표 기사 1개만 남깁니다. "
                        "모든 index는 반드시 하나의 event_group에 포함되어야 합니다. "
                        "반드시 JSON만 출력합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()

        logger.info("🧩 LLM 사건 그룹화 응답 수신")

        try:
            result = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("❌ LLM 사건 그룹화 JSON 파싱 실패")
            logger.error(content)
            return news_list[:limit]

        event_groups = result.get("event_groups", [])

        if not event_groups:
            logger.warning("⚠️ LLM 사건 그룹화 결과가 비어 있어 원본 선택 결과를 유지합니다.")
            return news_list[:limit]

        representative_indexes = []
        covered_indexes = set()

        for group in event_groups:
            indexes = group.get("indexes", [])
            representative_index = group.get("representative_index")

            normalized_indexes = []

            for value in indexes:
                try:
                    idx = int(value)
                except Exception:
                    continue

                if 1 <= idx <= len(news_list):
                    normalized_indexes.append(idx)
                    covered_indexes.add(idx)

            try:
                rep_idx = int(representative_index)
            except Exception:
                rep_idx = None

            # representative_index가 그룹 내부에 없거나 이상하면 그룹 첫 번째 기사로 대체
            if rep_idx not in normalized_indexes:
                rep_idx = normalized_indexes[0] if normalized_indexes else None

            if rep_idx and 1 <= rep_idx <= len(news_list):
                representative_indexes.append(rep_idx)

        # 혹시 LLM이 누락한 index가 있으면 독립 뉴스로 보고 유지
        all_indexes = set(range(1, len(news_list) + 1))
        missing_indexes = sorted(all_indexes - covered_indexes)

        if missing_indexes:
            logger.warning(f"⚠️ LLM 사건 그룹화에서 누락된 index 유지: {missing_indexes}")
            representative_indexes.extend(missing_indexes)

        # 중복 제거 + 원래 뉴스 순서 유지
        representative_index_set = set(representative_indexes)

        deduped_news = []

        for idx, news in enumerate(news_list, 1):
            if idx in representative_index_set:
                deduped_news.append(news)

        deduped_news = deduped_news[:limit]

        logger.info(
            f"🧠 LLM 사건 그룹화 중복 제거 완료: "
            f"{len(news_list)}개 → {len(deduped_news)}개"
        )

        for group in event_groups:
            indexes = group.get("indexes", [])
            representative_index = group.get("representative_index")

            if len(indexes) > 1:
                logger.info(
                    f"   🧩 사건 그룹: {group.get('event_group', '')} | "
                    f"대표 {representative_index} | "
                    f"그룹 {indexes}"
                )

        return deduped_news

    except Exception as e:
        logger.error(f"❌ LLM 사건 그룹화 중복 제거 실패: {e}")
        logger.warning("⚠️ 중복 제거 실패로 기존 선택 결과를 유지합니다.")
        return news_list[:limit]

def _supplement_after_dedup(
    selected_news: List[Dict],
    candidate_pool: List[Dict],
    used_indexes: set,
    limit: int
) -> List[Dict]:
    """
    사건 중복 제거 후 뉴스 수가 limit보다 적을 경우,
    후보군에서 아직 사용하지 않은 뉴스를 반복적으로 보충한다.

    핵심:
    - 한 번만 보충하지 않는다.
    - 보충 후 다시 사건 그룹화한다.
    - 그래도 부족하면 다음 후보 묶음을 또 보충한다.
    - limit에 도달하거나 후보가 소진될 때까지 반복한다.
    """
    if len(selected_news) >= limit:
        return selected_news[:limit]

    final_news = list(selected_news)
    batch_size = max(limit * 2, 20)

    while len(final_news) < limit:
        supplement_batch = []

        for idx, news in enumerate(candidate_pool):
            if idx in used_indexes:
                continue

            used_indexes.add(idx)

            prepared_news = _prepare_selected_news(
                news=news,
                importance_score=news.get("importance_score", 3),
                category=news.get("category") or "기타"
            )

            supplement_batch.append(prepared_news)

            if len(supplement_batch) >= batch_size:
                break

        if not supplement_batch:
            logger.warning("⚠️ 보충 가능한 후보가 더 이상 없습니다.")
            break

        before_count = len(final_news)

        combined_news = final_news + supplement_batch

        logger.info(
            f"➕ 부족분 반복 보충: 현재 {len(final_news)}개, "
            f"추가 후보 {len(supplement_batch)}개"
        )

        final_news = _deduplicate_by_llm_event_group(
            combined_news,
            limit=limit
        )

        after_count = len(final_news)

        logger.info(
            f"🔁 보충 후 사건 중복 제거 결과: "
            f"{before_count}개 → {after_count}개"
        )

        # 보충했는데도 개수가 전혀 늘지 않았고, 아직 후보가 남아있을 수 있으므로 계속 돈다.
        # 단, candidate_pool을 다 소진하면 위 supplement_batch가 비어서 종료된다.

    return final_news[:limit]


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
    logger.info(f"원본 후보 뉴스 수: {len(news_list)}개")
    logger.info(f"최종 선택 목표: {limit}개")
    logger.info(f"{'=' * 60}")

    # 1차 URL 중복 제거
    candidate_pool = _deduplicate_by_url(
        news_list,
        log_prefix="OpenAI 전달 전 후보"
    )

    if not candidate_pool:
        logger.warning("URL 중복 제거 후 남은 후보 뉴스가 없습니다.")
        return []

    logger.info(f"OpenAI 전달 후보 뉴스 수: {len(candidate_pool)}개")

    # 중복 제거 후에도 최종 limit개를 확보하기 위해
    # 1차 선별에서는 limit보다 넉넉하게 뽑는다.
    candidate_limit = min(len(candidate_pool), max(limit * 5, 50))

    candidate_text = _build_candidate_text(candidate_pool)

    prompt = f"""
아래는 네이버 뉴스 API로 수집한 뉴스 후보 목록입니다.

당신의 역할은 뉴스 편집자입니다.
아래 기준으로 최종 후보 뉴스 {candidate_limit}개 이하를 선택하세요.

[브리핑 이름]
{topic_name}

[선택해야 하는 뉴스 주제]
{topic_description}

[선택 기준]
1. 주제와 직접 관련 있는 뉴스만 선택하세요.
2. 기업 전략, 실적, 투자, 제휴, 정책, 규제, 기술 도입, 산업 변화에 영향이 큰 뉴스를 우선 선택하세요.
3. 홍보성 기사, 단순 행사 안내, 단순 제품 소개성 기사는 제외하세요.
4. 같은 사건처럼 보이는 기사가 여러 개 있어도, 이 단계에서는 판단이 애매하면 후보에 포함해도 됩니다.
5. 최종 후보로 최대 {candidate_limit}개까지 선택하세요.
6. 가능하면 서로 다른 사건, 서로 다른 기업, 서로 다른 정책, 서로 다른 기술 이슈가 골고루 포함되도록 선택하세요.
7. 명백히 주제와 무관한 기사는 제외하되, 주제 관련성이 어느 정도 있으면 후보에 포함하세요.
7. 이후 시스템이 같은 사건 중복을 한 번 더 제거합니다.

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
            temperature=0.1,
            max_tokens=1600
        )

        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info(f"🧾 뉴스 1차 선별 토큰 사용량: {tokens_used}")

        try:
            result = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("❌ OpenAI 1차 선별 응답 JSON 파싱 실패")
            logger.error(content)
            return _fallback_select(candidate_pool, limit)

        selected_items = result.get("selected", [])

        if not selected_items:
            logger.warning("⚠️ OpenAI가 선택한 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        selected_news = []
        used_indexes = set()

        for item in selected_items:
            try:
                selected_index = item.get("index")
                idx = int(selected_index) - 1
            except Exception:
                continue

            if idx < 0 or idx >= len(candidate_pool):
                continue

            news = candidate_pool[idx]

            news = _prepare_selected_news(
                news=news,
                importance_score=item.get("importance_score", 3),
                category=item.get("category") or "기타"
            )

            selected_news.append(news)
            used_indexes.add(idx)

            if len(selected_news) >= candidate_limit:
                break

        if not selected_news:
            logger.warning("⚠️ 유효하게 선별된 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        logger.info(f"✅ OpenAI 1차 뉴스 선별 완료: {len(selected_news)}개 후보 선택")

        # 2차 중복 제거:
        # 선택된 후보를 LLM이 같은 사건 기준으로 그룹화하여 중복 제거한다.
        deduped_selected_news = _deduplicate_by_llm_event_group(
            selected_news,
            limit=limit
        )

        if not deduped_selected_news:
            logger.warning("⚠️ 사건 중복 제거 후 남은 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        # 중복 제거 후 limit보다 부족하면 후보군에서 추가 보충
        if len(deduped_selected_news) < limit:
            deduped_selected_news = _supplement_after_dedup(
                selected_news=deduped_selected_news,
                candidate_pool=candidate_pool,
                used_indexes=used_indexes,
                limit=limit
            )

        final_news = deduped_selected_news[:limit]

        logger.info(f"✅ 최종 뉴스 선별 완료: {len(final_news)}개 선택")

        for i, news in enumerate(final_news, 1):
            logger.info(
                f"   [{i}] "
                f"중요도 {news.get('importance_score', 3)} | "
                f"{news.get('category', '기타')} | "
                f"{news.get('source', '언론사 미상')} | "
                f"{news.get('title', '')[:60]}"
            )

        return final_news

    except Exception as e:
        logger.error(f"❌ OpenAI 뉴스 선별 실패: {e}")
        logger.warning(f"⚠️ 실패 시 URL 중복 제거 후 후보 뉴스 앞에서 {limit}개 사용")
        return _fallback_select(news_list, limit)