"""
이메일 자동 발송 모듈
"""
import smtplib
import os
import html as html_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime, formataddr
from datetime import datetime
from dotenv import load_dotenv
import logging
import pytz

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================
# 이메일 설정
# ====================================
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# 기본 사내용 수신자
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# 추가 브리핑 수신자 예시. 실제 사용 여부는 config의 receiver_env가 결정한다.
EMAIL_RECEIVER_ECONOMY = os.getenv("EMAIL_RECEIVER_ECONOMY")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ====================================
# OpenAI 설정
# ====================================
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


def safe_text(value):
    """
    HTML 표시용 문자열 안전 변환
    """
    if value is None:
        return ""
    return html_lib.escape(str(value).strip())


def safe_url(value):
    """
    링크 URL 안전 변환
    """
    if value is None:
        return "#"
    return html_lib.escape(str(value).strip(), quote=True)


def safe_int(value, default=3, min_value=1, max_value=5):
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


def safe_count(value, default=0):
    """
    대시보드 숫자 안전 변환
    """
    try:
        return int(value)
    except Exception:
        return default


def format_korean_datetime(date_string):
    """
    날짜 문자열을 한국식 표현으로 변환한다.

    지원 형식:
    - 네이버 pubDate: Tue, 13 May 2026 09:01:00 +0900
    - ISO/KST: 2026-05-13T09:01:00+09:00
    """
    raw_value = str(date_string or "").strip()

    if not raw_value:
        return ""

    try:
        try:
            dt = parsedate_to_datetime(raw_value)
        except Exception:
            dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))

        kst = pytz.timezone("Asia/Seoul")

        if dt.tzinfo is None:
            dt = kst.localize(dt)

        dt_kst = dt.astimezone(kst)

        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        weekday = weekdays[dt_kst.weekday()]

        hour = dt_kst.hour
        if hour < 12:
            period = "오전"
            display_hour = hour if hour > 0 else 12
        else:
            period = "오후"
            display_hour = hour if hour == 12 else hour - 12

        return (
            f"{dt_kst.year}년 {dt_kst.month:02d}월 {dt_kst.day:02d}일 "
            f"{weekday} {period} {display_hour:02d}:{dt_kst.minute:02d}"
        )

    except Exception:
        return safe_text(raw_value)


def get_today_date_text():
    """
    오늘 날짜 문자열 생성
    예: 2026년 05월 08일 금요일
    """
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    weekday = weekdays[now_kst.weekday()]

    formatted_date = f"{now_kst.year}년 {now_kst.month:02d}월 {now_kst.day:02d}일 {weekday}"

    return formatted_date, now_kst


def determine_receiver_env_name(
    briefing_name=None,
    subject_prefix=None,
    section_results=None,
    receiver_env_name=None
):
    """
    사용할 수신자 환경변수명 결정.

    원칙:
    - 브리핑 종류를 코드에서 특정 단어로 추정하지 않는다.
    - config의 receiver_env 값이 main.py를 통해 receiver_env_name으로 전달되면 그 값을 그대로 사용한다.
    - 별도 지정이 없을 때만 기본 EMAIL_RECEIVER를 사용한다.
    """
    if receiver_env_name:
        return receiver_env_name

    return "EMAIL_RECEIVER"


def get_receiver_list(receiver_env_name=None):
    """
    환경변수에서 수신자 목록 가져오기

    지원 형식:
    1. 이메일만
       EMAIL_RECEIVER=seonbin.kim@lotte.net,wootak.ko@lotte.net

    2. 이름|이메일
       EMAIL_RECEIVER=김선빈|seonbin.kim@lotte.net,고우탁|wootak.ko@lotte.net

    Returns:
        [
            {
                "name": "김선빈",
                "email": "seonbin.kim@lotte.net",
                "display": "=?utf-8?b?...?= <seonbin.kim@lotte.net>"
            },
            ...
        ]
    """
    if receiver_env_name is None:
        receiver_env_name = "EMAIL_RECEIVER"

    receiver_value = os.getenv(receiver_env_name)

    if not receiver_value:
        return []

    receivers = []

    for raw_item in receiver_value.split(","):
        item = raw_item.strip()

        if not item:
            continue

        if "|" in item:
            name, email = item.split("|", 1)
            name = name.strip()
            email = email.strip()
        else:
            name = ""
            email = item.strip()

        if not email:
            continue

        display = formataddr((name, email)) if name else email

        receivers.append({
            "name": name,
            "email": email,
            "display": display
        })

    return receivers


def build_section_dashboard(section_result):
    """
    섹션 제목 아래, 핵심요약 3줄 위에 표시할 간단 대시보드.

    표시 항목:
    - 전체검색수: 네이버 API 검색 결과로 확인한 전체 기사 수
    - 24시간초과제외: recent_hours 기준을 벗어나 제외된 기사 수
    - 코드규칙제외: URL 중복, 최근 발송 이슈, 제외 키워드, 저품질/사진성 그룹 등
      AI 호출 전에 코드 규칙으로 제외한 기사 수
    - AI 중복제외: AI가 고른 최종 후보 안에서 코드 규칙으로 다시 제거한 중복 기사 수
      예: AI가 10개를 골랐고 최종 중복 제거로 2개가 빠지면 2로 표시한다.
    - AI선별: 최종 중복 제거 후 실제 메일 요약 대상으로 남은 기사 수
    """
    if not section_result:
        return ""

    summaries = section_result.get("summaries", []) or []
    scrape_stats = section_result.get("scrape_stats", {}) or {}

    total_seen_count = safe_count(scrape_stats.get("total_seen_count", 0))
    old_news_count = safe_count(scrape_stats.get("old_news_count", 0))

    code_rule_excluded_count = safe_count(scrape_stats.get("code_rule_excluded_count", None), None)
    if code_rule_excluded_count is None:
        code_rule_excluded_count = (
            safe_count(scrape_stats.get("duplicate_count", 0))
            + safe_count(scrape_stats.get("issue_filter_excluded_count", 0))
            + safe_count(scrape_stats.get("exclude_keyword_excluded_count", 0))
            + safe_count(scrape_stats.get("grouping_low_quality_article_count", 0))
        )

    ai_duplicate_excluded_count = safe_count(scrape_stats.get("ai_duplicate_excluded_count", None), None)
    if ai_duplicate_excluded_count is None:
        ai_duplicate_excluded_count = 0

    selected_count = safe_count(section_result.get("selected_count", len(summaries)))

    return f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="border-collapse:collapse; margin:0 0 12px 0;">
            <tr>
                <td style="padding:0 0 10px 0; border-bottom:1px solid #dddddd;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="border-collapse:collapse; width:100%;">
                        <tr>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    전체검색수
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {total_seen_count}
                                </div>
                            </td>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    24시간초과제외
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {old_news_count}
                                </div>
                            </td>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    코드규칙제외
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {code_rule_excluded_count}
                                </div>
                            </td>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    AI 중복제외
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {ai_duplicate_excluded_count}
                                </div>
                            </td>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    AI선별
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {selected_count}
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    """

def build_section_insights(section_title, summaries, scrape_stats=None):
    """
    섹션별 핵심 3줄 생성
    """
    if not summaries:
        return """
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                   style="border-collapse:collapse; margin:0 0 14px 0;">
                <tr>
                    <td style="padding:0 0 10px 0; border-bottom:1px solid #dddddd;">
                        <div style="font-size:13px; line-height:1.6;">
                            1. 이 섹션에서 요약할 뉴스가 없습니다.
                        </div>
                    </td>
                </tr>
            </table>
        """

    news_text_list = []

    for i, news in enumerate(summaries, 1):
        title = str(news.get("title", "")).strip()
        summary = str(news.get("summary", "")).strip()
        category = str(news.get("category", "기타")).strip()
        source = str(news.get("source", "언론사 미상")).strip()
        importance_score = str(news.get("importance_score", "3")).strip()

        news_text_list.append(
            f"{i}. [{category} / {source} / 중요도 {importance_score}] {title}\n요약: {summary}"
        )

    news_text = "\n\n".join(news_text_list)

    prompt = f"""
아래는 "{section_title}" 섹션에 들어갈 뉴스 요약 목록입니다.

이 섹션의 뉴스들만 보고, 메일 상단에 넣을 핵심 3줄을 작성하세요.

[작성 기준]
1. 뉴스 개수, 최상위 뉴스, 중요도 개수 같은 메타 정보는 쓰지 마세요.
2. 이 섹션 뉴스 전체를 관통하는 흐름을 요약하세요.
3. 각 줄은 너무 길지 않게 작성하세요.
4. 반드시 정확히 3줄만 작성하세요.
5. 각 줄은 "1. ", "2. ", "3. "으로 시작하세요.
6. 기사 요약에 없는 사실은 추가하지 마세요.
7. 추측하지 말고, 제공된 뉴스 요약 안에서만 정리하세요.

[뉴스 목록]
{news_text}
"""

    try:
        if client is None:
            raise ValueError("OPENAI_API_KEY가 없거나 OpenAI 클라이언트를 초기화할 수 없습니다.")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 목록 전체를 보고 해당 섹션의 핵심 흐름을 3줄로 요약합니다. "
                        "원문 요약에 없는 사실은 추가하지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=350
        )

        insight_text = response.choices[0].message.content.strip()
        insight_tokens = response.usage.total_tokens if response.usage else 0
        if isinstance(scrape_stats, dict):
            scrape_stats["insight_tokens"] = safe_count(scrape_stats.get("insight_tokens", 0)) + insight_tokens
        logger.info(f"🧾 [{section_title}] 메일 핵심 3줄 토큰 사용량: {insight_tokens}")
        lines = [line.strip() for line in insight_text.split("\n") if line.strip()]
        lines = lines[:3]

        if not lines:
            raise ValueError("핵심 3줄 응답이 비어 있습니다.")

    except Exception as e:
        logger.error(f"❌ [{section_title}] 핵심 3줄 생성 실패: {e}")

        lines = []
        for i, news in enumerate(summaries[:3], 1):
            summary = str(news.get("summary", "")).strip()
            if summary:
                lines.append(f"{i}. {summary}")

        if not lines:
            lines = ["1. 이 섹션의 핵심 요약을 생성하지 못했습니다."]

    html_lines = ""

    for line in lines[:3]:
        html_lines += f"""
            <div style="font-size:13px; line-height:1.65; margin:0 0 3px 0; word-break:keep-all;">
                {safe_text(line)}
            </div>
        """

    return f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="border-collapse:collapse; margin:0 0 15px 0;">
            <tr>
                <td style="padding:0 0 11px 0; border-bottom:1px solid #dddddd;">
                    {html_lines}
                </td>
            </tr>
        </table>
    """


def get_section_color(index):
    """
    섹션별 포인트 색상 반환
    네이버메일 호환을 위해 class가 아니라 inline style에서 사용한다.
    """
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#7c3aed",
        "#ea580c"
    ]

    return colors[index % len(colors)]


def build_news_section(section_result, section_index):
    """
    뉴스 섹션 HTML 생성.

    구조:
    - 섹션 제목
    - 섹션별 간단 대시보드
    - 핵심요약 3줄
    - 뉴스 목록
    """
    section_title = section_result.get("section_name", f"뉴스 섹션 {section_index + 1}")
    summaries = section_result.get("summaries", []) or []
    section_color = get_section_color(section_index)

    html_body = f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="border-collapse:collapse; margin:0 0 22px 0;">
            <tr>
                <td style="border-left:4px solid {section_color}; padding:0 0 0 12px;">
                    <div style="font-size:17px; font-weight:900; line-height:1.35; color:{section_color}; margin:0 0 10px 0;">
                        {safe_text(section_title)}
                    </div>
    """

    html_body += build_section_dashboard(section_result)

    html_body += build_section_insights(
        section_title=section_title,
        summaries=summaries,
        scrape_stats=section_result.get("scrape_stats", {})
    )

    if not summaries:
        html_body += """
                    <div style="font-size:13px; line-height:1.5; margin:0 0 12px 0;">
                        표시할 뉴스가 없습니다.
                    </div>
                </td>
            </tr>
        </table>
        """
        return html_body

    for i, news in enumerate(summaries, 1):
        published_date = format_korean_datetime(
            news.get("published_at") or news.get("published_at_kst") or ""
        )

        title = safe_text(news.get("title", "제목 없음"))
        url = safe_url(news.get("url", "#"))
        summary = news.get("summary", "")
        summary_html = safe_text(summary).replace("\n", "<br>")

        category = safe_text(news.get("category", section_title) or section_title)
        source = safe_text(news.get("source", "언론사 미상") or "언론사 미상")
        importance_score = safe_int(news.get("importance_score", 3))
        stars = "★" * importance_score + "☆" * (5 - importance_score)

        html_body += f"""
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="border-collapse:collapse; margin:0 0 16px 0; border-bottom:1px solid #dddddd;">
                        <tr>
                            <td style="padding:0 0 13px 0;">
                                <div style="margin:0 0 5px 0;">
                                    <a href="{url}" target="_blank"
                                       style="font-size:15px; font-weight:800; line-height:1.42; text-decoration:none;">
                                        {i}. {title}
                                    </a>
                                </div>

                                <div style="font-size:11px; line-height:1.5; margin:0 0 7px 0;">
                                    <span style="font-weight:800;">{category}</span>
                                    <span>　</span>
                                    <span style="font-weight:800;">{source}</span>
                                    <span>　</span>
                                    <span style="font-weight:800; color:#ea580c;">중요도 {importance_score}</span>
                                    <span>　</span>
                                    <span style="color:#f59e0b; letter-spacing:-1px;">{stars}</span>
                                    <span>　</span>
                                    <span>{published_date}</span>
                                </div>

                                <div style="font-size:13px; line-height:1.65; margin:0; padding:0; word-break:keep-all;">
                                    {summary_html}
                                </div>
                            </td>
                        </tr>
                    </table>
        """

    html_body += """
                </td>
            </tr>
        </table>
    """

    return html_body


def normalize_section_results(section_results=None, summaries=None):
    """
    section_results 구조 정규화
    """
    if section_results is not None:
        return section_results

    if summaries is not None:
        return [
            {
                "section_name": "뉴스 브리핑",
                "summaries": summaries,
                "raw_count": len(summaries),
                "selected_count": len(summaries),
                "scrape_stats": {
                    "total_seen_count": len(summaries),
                    "duplicate_count": 0,
                    "old_news_count": 0
                }
            }
        ]

    return []


def create_html_email(
    briefing_name=None,
    subject_prefix=None,
    section_results=None,
    summaries=None
):
    """
    HTML 이메일 생성
    네이버메일 호환을 위해 table layout + inline style 중심으로 작성한다.
    """
    if briefing_name is None:
        briefing_name = "뉴스 브리핑"

    if subject_prefix is None:
        subject_prefix = briefing_name

    section_results = normalize_section_results(
        section_results=section_results,
        summaries=summaries
    )

    today_text, now_kst = get_today_date_text()
    mail_title = f"{today_text} - {subject_prefix}"

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{safe_text(mail_title)}</title>
</head>
<body style="margin:0; padding:0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
           style="border-collapse:collapse; width:100%;">
        <tr>
            <td align="left" style="padding:14px 12px;">
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                       style="border-collapse:collapse; width:100%; max-width:900px;">
                    <tr>
                        <td style="padding:0 0 12px 0; border-bottom:2px solid #999999;">
                            <div style="font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;
                                        font-size:22px; font-weight:900; line-height:1.35;
                                        letter-spacing:-0.5px;">
                                {safe_text(mail_title)}
                            </div>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:16px 0 0 0;
                                   font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;">
"""

    for index, section_result in enumerate(section_results):
        html_body += build_news_section(
            section_result=section_result,
            section_index=index
        )

    html_body += f"""
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:8px 0 0 0;
                                   font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;
                                   color:#a1a1aa; font-size:11px; line-height:1.5;">
                            <div style="margin:0 0 3px 0;">
                                <strong>이 메일은 AI를 통하여 자동으로 선별, 요약되어 발송되었습니다.</strong>
                            </div>
                            <div style="margin:0 0 3px 0;">
                                AI News Scraper v1.0
                            </div>
                            <div style="margin:0;">
                                © {now_kst.year} AI News Scraper. All rights reserved.
                            </div>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    return html_body


def send_email(
    summaries=None,
    subject=None,
    briefing_name=None,
    subject_prefix=None,
    section_results=None,
    receiver_env_name=None
):
    """
    이메일로 뉴스 요약 발송

    수신자 형식:
    EMAIL_RECEIVER=김선빈|seonbin.kim@lotte.net,고우탁|wootak.ko@lotte.net
    """
    try:
        if not EMAIL_SENDER:
            return {
                "success": False,
                "message": "EMAIL_SENDER 환경변수가 설정되지 않았습니다."
            }

        if not EMAIL_PASSWORD:
            return {
                "success": False,
                "message": "EMAIL_PASSWORD 환경변수가 설정되지 않았습니다."
            }

        section_results = normalize_section_results(
            section_results=section_results,
            summaries=summaries
        )

        if briefing_name is None:
            briefing_name = "뉴스 브리핑"

        if subject_prefix is None:
            subject_prefix = briefing_name

        selected_receiver_env_name = determine_receiver_env_name(
            briefing_name=briefing_name,
            subject_prefix=subject_prefix,
            section_results=section_results,
            receiver_env_name=receiver_env_name
        )

        receiver_list = get_receiver_list(selected_receiver_env_name)

        if not receiver_list:
            logger.error(f"❌ 수신자 이메일이 설정되지 않았습니다: {selected_receiver_env_name}")
            return {
                "success": False,
                "message": f"{selected_receiver_env_name} 환경변수를 확인하세요"
            }

        if subject is None:
            today_text, _ = get_today_date_text()
            subject = f"📰 {today_text} - {subject_prefix}"

        html_body = create_html_email(
            briefing_name=briefing_name,
            subject_prefix=subject_prefix,
            section_results=section_results
        )

        success_count = 0
        failed_list = []

        logger.info(f"📧 이메일 개별 발송 시작: {len(receiver_list)}명")
        logger.info(f"📮 사용 수신자 환경변수: {selected_receiver_env_name}")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)

            for receiver_info in receiver_list:
                receiver_name = ""
                receiver_email = ""
                receiver_display = ""

                try:
                    if isinstance(receiver_info, dict):
                        receiver_name = receiver_info.get("name") or ""
                        receiver_email = receiver_info.get("email") or ""
                        receiver_display = receiver_info.get("display") or receiver_email
                    else:
                        raw_receiver = str(receiver_info).strip()

                        if "|" in raw_receiver:
                            receiver_name, receiver_email = raw_receiver.split("|", 1)
                            receiver_name = receiver_name.strip()
                            receiver_email = receiver_email.strip()
                            receiver_display = formataddr((receiver_name, receiver_email))
                        else:
                            receiver_email = raw_receiver
                            receiver_display = receiver_email
                            receiver_name = receiver_email

                    if not receiver_email:
                        raise ValueError("수신자 이메일이 비어 있습니다.")

                    if not receiver_name:
                        receiver_name = receiver_email

                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"] = EMAIL_SENDER
                    msg["To"] = receiver_display

                    plain_body = f"{subject}\n\nHTML 메일을 지원하는 환경에서 뉴스 브리핑을 확인해 주세요."

                    text_part = MIMEText(plain_body, "plain", "utf-8")
                    html_part = MIMEText(html_body, "html", "utf-8")

                    msg.attach(text_part)
                    msg.attach(html_part)

                    server.send_message(msg)
                    logger.info(f"   ✅ {receiver_name} ({receiver_email}) 발송 완료")
                    success_count += 1

                except Exception as e:
                    display_name = receiver_name or receiver_email or str(receiver_info)
                    logger.error(f"   ❌ {display_name} 발송 실패: {str(e)}")
                    failed_list.append(display_name)

        if success_count == len(receiver_list):
            logger.info(f"🎉 모든 이메일 발송 완료! ({success_count}명)")
            return {
                "success": True,
                "message": f"{success_count}명에게 개별 발송 완료"
            }

        if success_count > 0:
            logger.warning(f"⚠️ 일부 발송 완료: {success_count}/{len(receiver_list)}명")
            return {
                "success": True,
                "message": f"{success_count}명 성공, {len(failed_list)}명 실패: {', '.join(failed_list)}"
            }

        logger.error("❌ 모든 이메일 발송 실패")
        return {
            "success": False,
            "message": f"모든 발송 실패: {', '.join(failed_list)}"
        }

    except smtplib.SMTPAuthenticationError:
        logger.error("❌ 로그인 실패! 이메일/비밀번호를 확인하세요.")
        return {
            "success": False,
            "message": "SMTP 인증 실패. Gmail 앱 비밀번호 확인 필요"
        }

    except Exception as e:
        logger.error(f"❌ 이메일 발송 실패: {str(e)}")
        return {
            "success": False,
            "message": f"이메일 발송 실패: {str(e)}"
        }


def send_test_email(receiver_env_name="EMAIL_RECEIVER"):
    """
    이메일 전송만 테스트한다.

    - 뉴스 수집 안 함
    - 뉴스 요약 안 함
    - OpenAI 호출 안 함
    - SMTP 로그인/전송/수신자 파싱만 확인
    - 섹션별 대시보드 위치 확인 가능
    """
    today_text, _ = get_today_date_text()

    test_section_results = [
        {
            "section_name": "경제 뉴스 브리핑",
            "summaries": [
                {
                    "title": "테스트 경제 뉴스 제목입니다",
                    "summary": "경제 뉴스 요약 테스트 문장입니다.",
                    "url": "#",
                    "published_at": "",
                    "importance_score": 4,
                    "category": "경제",
                    "source": "테스트언론"
                }
            ],
            "raw_count": 100,
            "selected_count": 10,
            "scrape_stats": {
                "total_seen_count": 1291,
                "duplicate_count": 760,
                "old_news_count": 431,
                "issue_filter_excluded_count": 12,
                "exclude_keyword_excluded_count": 8,
                "grouping_low_quality_article_count": 5,
                "grouping_duplicate_article_count": 38,
                "code_rule_excluded_count": 785,
                "ai_duplicate_excluded_count": 38,
                "final_candidate_count": 100
            }
        },
        {
            "section_name": "부동산 뉴스 브리핑",
            "summaries": [
                {
                    "title": "테스트 부동산 뉴스 제목입니다",
                    "summary": "부동산 뉴스 요약 테스트 문장입니다.",
                    "url": "#",
                    "published_at": "",
                    "importance_score": 5,
                    "category": "부동산",
                    "source": "테스트언론"
                }
            ],
            "raw_count": 100,
            "selected_count": 10,
            "scrape_stats": {
                "total_seen_count": 868,
                "duplicate_count": 688,
                "old_news_count": 80,
                "issue_filter_excluded_count": 4,
                "exclude_keyword_excluded_count": 18,
                "grouping_low_quality_article_count": 3,
                "grouping_duplicate_article_count": 14,
                "code_rule_excluded_count": 713,
                "ai_duplicate_excluded_count": 14,
                "final_candidate_count": 100
            }
        }
    ]

    return send_email(
        subject=f"✅ 이메일 전송 테스트 - {today_text}",
        briefing_name="이메일 전송 테스트",
        subject_prefix="이메일 전송 테스트",
        section_results=test_section_results,
        receiver_env_name=receiver_env_name
    )


if __name__ == "__main__":
    result = send_test_email(receiver_env_name="EMAIL_RECEIVER")
    logger.info(result)