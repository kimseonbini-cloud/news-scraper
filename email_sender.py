# =============================================================================
# [파일 설명]
# - 수행 기능: 뉴스 요약 결과를 HTML 이메일로 구성하고 SMTP를 통해 수신자에게 발송합니다.
# - 프로세스: 수신자/발송 방식 정규화 -> 섹션 HTML 구성 -> 메일 MIME 생성 -> SMTP 로그인/발송 -> 성공 여부 반환
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: section_results 또는 summaries, 제목/브리핑명, 수신자 환경변수, SMTP 환경변수
# - 리턴값/출력: send_email()은 발송 성공 여부 bool을 반환하고, HTML 생성 함수들은 문자열을 반환합니다.
# =============================================================================

"""
이메일 자동 발송 모듈
"""
import smtplib
import os
import html as html_lib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime, formataddr
from datetime import datetime
from dotenv import load_dotenv
from openai_usage import (
    create_chat_completion as create_openai_chat_completion,
    record_openai_usage,
    openai_token_limit_kwargs,
    openai_temperature_kwargs,
    openai_reasoning_effort_kwargs,
    is_gpt5_model,
)
import logging
import pytz

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # OpenAI클래스

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈로거

# ====================================
# 이메일 설정
# ====================================
EMAIL_SENDER = os.getenv("EMAIL_SENDER")                                      # 발신자이메일주소
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")                                  # SMTP앱비밀번호

# 기본 사내용 수신자
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")                                  # 기본수신자환경값

# 추가 브리핑 수신자 예시. 실제 사용 여부는 config의 receiver_env가 결정한다.
EMAIL_RECEIVER_ECONOMY = os.getenv("EMAIL_RECEIVER_ECONOMY")                  # 경제브리핑수신자환경값

SMTP_SERVER = "smtp.gmail.com"                                                # SMTP서버주소
SMTP_PORT = 587                                                               # SMTPTLS포트

# ====================================
# OpenAI 설정
# ====================================
MODEL = os.getenv("EMAIL_INSIGHT_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))  # 메일3줄요약모델명
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                                  # OpenAI인증키
EMAIL_INSIGHT_USE_AI_DEFAULT = os.getenv("EMAIL_INSIGHT_USE_AI", "true").lower() not in {"0", "false", "no", "off"}  # 메일3줄AI요약기본값
EMAIL_INSIGHT_MAX_COMPLETION_TOKENS = int(os.getenv("EMAIL_INSIGHT_MAX_COMPLETION_TOKENS", "2400"))  # 메일3줄요약응답토큰상한
EMAIL_INSIGHT_RETRY_MAX_COMPLETION_TOKENS = int(os.getenv("EMAIL_INSIGHT_RETRY_MAX_COMPLETION_TOKENS", "3600"))  # 메일3줄요약재시도토큰상한

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)  # OpenAI클라이언트
else:
    client = None  # OpenAI클라이언트


# [코드 이해 주석]
# - 역할: HTML 표시용 문자열 안전 변환.
# - 호출하는 곳: email_sender.build_news_section, email_sender.build_related_reports_html,
# email_sender.build_section_insights, email_sender.create_html_email, email_sender.format_korean_datetime
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def safe_text(value):
    """
    HTML 표시용 문자열 안전 변환
    """
    if value is None:
        return ""
    return html_lib.escape(str(value).strip())


# [코드 이해 주석]
# - 역할: 링크 URL 안전 변환.
# - 호출하는 곳: email_sender.build_news_section, email_sender.build_related_reports_html
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def safe_url(value):
    """
    링크 URL 안전 변환
    """
    if value is None:
        return "#"
    return html_lib.escape(str(value).strip(), quote=True)


# [코드 이해 주석]
# - 역할: 중요도 점수 안전 변환.
# - 호출하는 곳: email_sender.build_news_section, email_sender.build_section_insights
# - 파라미터: value: Any, default: Any = 3, min_value: Any = 1, max_value: Any = 5
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def safe_int(value, default=3, min_value=1, max_value=5):
    """
    중요도 점수 안전 변환
    """
    try:
        number = int(value)  # 숫자값
    except Exception:
        number = default  # 숫자값

    if number < min_value:
        return min_value

    if number > max_value:
        return max_value

    return number


# [코드 이해 주석]
# - 역할: 대시보드 숫자 안전 변환.
# - 호출하는 곳: email_sender.build_news_section, email_sender.build_related_reports_html,
# email_sender.build_section_dashboard, email_sender.build_section_insights
# - 파라미터: value: Any, default: Any = 0
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def safe_count(value, default=0):
    """
    대시보드 숫자 안전 변환
    """
    try:
        return int(value)
    except Exception:
        return default


# [코드 이해 주석]
# - 역할: 입력값을 화면 표시나 후속 처리에 안전한 형태로 변환하는 보조 함수입니다.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: value: Any, default: Any = False
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def safe_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()  # 텍스트
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


# [코드 이해 주석]
# - 역할: 날짜 문자열을 한국식 표현으로 변환한다.
# - 호출하는 곳: email_sender.build_news_section
# - 파라미터: date_string: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def format_korean_datetime(date_string):
    """
    날짜 문자열을 한국식 표현으로 변환한다.

    지원 형식:
    - 네이버 pubDate: Tue, 13 May 2026 09:01:00 +0900
    - ISO/KST: 2026-05-13T09:01:00+09:00
    """
    raw_value = str(date_string or "").strip()  # 원본값

    if not raw_value:
        return ""

    try:
        try:
            dt = parsedate_to_datetime(raw_value)  # 일시
        except Exception:
            dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))  # 일시

        kst = pytz.timezone("Asia/Seoul")  # kst

        if dt.tzinfo is None:
            dt = kst.localize(dt)  # 일시

        dt_kst = dt.astimezone(kst)  # 한국시간일시

        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]  # weekdays
        weekday = weekdays[dt_kst.weekday()]  # weekday

        hour = dt_kst.hour  # hour
        if hour < 12:
            period = "오전"  # period
            display_hour = hour if hour > 0 else 12  # 표시건수시각
        else:
            period = "오후"  # period
            display_hour = hour if hour == 12 else hour - 12  # 표시건수시각

        return (
            f"{dt_kst.year}년 {dt_kst.month:02d}월 {dt_kst.day:02d}일 "
            f"{weekday} {period} {display_hour:02d}:{dt_kst.minute:02d}"
        )

    except Exception:
        return safe_text(raw_value)


# [코드 이해 주석]
# - 역할: 오늘 날짜 문자열 생성.
# - 호출하는 곳: email_sender.create_html_email, email_sender.send_email, email_sender.send_test_email
# - 파라미터: 없음
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_today_date_text():
    """
    오늘 날짜 문자열 생성
    예: 2026년 05월 08일 금요일
    """
    kst = pytz.timezone("Asia/Seoul")  # kst
    now_kst = datetime.now(kst)  # 현재한국시간

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]  # weekdays
    weekday = weekdays[now_kst.weekday()]  # weekday

    formatted_date = f"{now_kst.year}년 {now_kst.month:02d}월 {now_kst.day:02d}일 {weekday}"  # 형식화날짜날짜

    return formatted_date, now_kst


# [코드 이해 주석]
# - 역할: 사용할 수신자 환경변수명 결정.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: briefing_name: Any = None, subject_prefix: Any = None, section_results: Any = None, receiver_env_name: Any =
# None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def determine_receiver_env_name(
    briefing_name=None,  # briefing이름
    subject_prefix=None,  # 메일제목prefix
    section_results=None,  # 섹션결과목록
    receiver_env_name=None  # 수신자env이름
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


# [코드 이해 주석]
# - 역할: 환경변수에서 수신자 목록 가져오기.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: receiver_env_name: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
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
        receiver_env_name = "EMAIL_RECEIVER"  # 수신자env이름

    receiver_value = os.getenv(receiver_env_name)  # 수신자값

    if not receiver_value:
        return []

    receivers = []  # 수신자목록

    for raw_item in receiver_value.split(","):  # 원본항목
        item = raw_item.strip()  # 항목

        if not item:
            continue

        if "|" in item:
            name, email = item.split("|", 1)  # 이름,이메일주소
            name = name.strip()  # 이름
            email = email.strip()  # 이메일주소
        else:
            name = ""  # 이름
            email = item.strip()  # 이메일주소

        if not email:
            continue

        display = formataddr((name, email)) if name else email  # 표시건수

        receivers.append({
            "name": name,
            "email": email,
            "display": display
        })

    return receivers


# [코드 이해 주석]
# - 역할: 이메일 발송 방식 변환.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: value: Any, default: Any = 'individual'
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_email_send_mode(value, default="individual"):
    """
    이메일 발송 방식 변환.

    - individual: 수신자별 개별 발송
    - bulk: 전체 수신자를 To에 넣어 한 번에 발송
    """
    default = str(default or "individual").strip().lower()  # 기본값
    if default not in {"individual", "bulk"}:
        default = "individual"  # 기본값

    text = str(value or default).strip().lower()  # 텍스트
    if text in {"bulk", "all", "group", "combined", "single", "전체", "전체발송"}:
        return "bulk"
    if text in {"individual", "separate", "each", "personal", "개별", "개별발송"}:
        return "individual"

    logger.warning("⚠️ 알 수 없는 이메일 발송 방식입니다: %s. 기본값 %s를 사용합니다.", value, default)
    return default


# [코드 이해 주석]
# - 역할: dict 또는 문자열 수신자 정보를 {name, email, display} 구조로 변환한다.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: receiver_info: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_receiver_info(receiver_info):
    """
    dict 또는 문자열 수신자 정보를 {name, email, display} 구조로 변환한다.
    """
    receiver_name = ""  # 수신자이름
    receiver_email = ""  # 수신자이메일주소
    receiver_display = ""  # 수신자표시건수

    if isinstance(receiver_info, dict):
        receiver_name = str(receiver_info.get("name") or "").strip()  # 수신자이름
        receiver_email = str(receiver_info.get("email") or "").strip()  # 수신자이메일주소
        receiver_display = str(receiver_info.get("display") or "").strip()  # 수신자표시건수
    else:
        raw_receiver = str(receiver_info or "").strip()  # 원본수신자

        if "|" in raw_receiver:
            receiver_name, receiver_email = raw_receiver.split("|", 1)  # 수신자이름,수신자이메일주소
            receiver_name = receiver_name.strip()  # 수신자이름
            receiver_email = receiver_email.strip()  # 수신자이메일주소
        else:
            receiver_email = raw_receiver  # 수신자이메일주소
            receiver_name = receiver_email  # 수신자이름

    if not receiver_email:
        raise ValueError("수신자 이메일이 비어 있습니다.")

    if not receiver_name:
        receiver_name = receiver_email  # 수신자이름

    if not receiver_display:
        receiver_display = formataddr((receiver_name, receiver_email)) if receiver_name else receiver_email  # 수신자표시건수

    return {
        "name": receiver_name,
        "email": receiver_email,
        "display": receiver_display,
    }


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: subject: Any, html_body: Any, to_header: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_email_message(subject, html_body, to_header):
    msg = MIMEMultipart("alternative")  # msg
    msg["Subject"] = subject  # 처리값
    msg["From"] = EMAIL_SENDER  # msgFrom
    msg["To"] = to_header  # msgTo

    plain_body = f"{subject}\n\nHTML 메일을 지원하는 환경에서 뉴스 브리핑을 확인해 주세요."  # 일반텍스트본문

    text_part = MIMEText(plain_body, "plain", "utf-8")  # 텍스트part
    html_part = MIMEText(html_body, "html", "utf-8")  # HTMLpart

    msg.attach(text_part)
    msg.attach(html_part)

    return msg


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화하는 내부 보조 함수입니다.
# - 호출하는 곳: email_sender.build_related_reports_html
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def _normalize_url_for_compare(value):
    text = str(value or "").strip().lower()  # 텍스트
    if not text:
        return ""
    text = re.sub(r"^https?://", "", text)  # 텍스트
    text = re.sub(r"^www\.", "", text)  # 텍스트
    return text.rstrip("/")


# [코드 이해 주석]
# - 역할: 관련보도 제목/링크 목록 HTML 생성.
# - 호출하는 곳: email_sender.build_news_section
# - 파라미터: news: Any, toggle_id: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_related_reports_html(news, toggle_id=None):
    """
    관련보도 제목/링크 목록 HTML 생성.

    메일 내부 토글은 클라이언트별 지원이 불안정하므로, 생성된
    GitHub Pages 상세 페이지 링크를 우선 사용한다. 상세 페이지 URL이
    없을 때만 상위 3건을 fallback으로 표시한다.
    """
    related_url = str(news.get("related_reports_url") or "").strip()  # relatedURL
    related_count_from_page = safe_count(news.get("related_reports_count"), 0)  # 관련보도페이지기준건수
    # 1) 관련보도 상세 페이지 URL이 있으면 메일 안에는 링크만 노출한다.
    #    메일 클라이언트마다 접기/펼치기 HTML 지원이 달라, 긴 관련 기사 목록은 GitHub Pages 상세 페이지로 보낸다.
    if related_url and related_url != "#":
        related_count_text = related_count_from_page or safe_count(news.get("group_article_count"), 0)  # 관련보도표시건수
        if related_count_text <= 0:
            related_count_text = "전체"  # 관련보도표시건수
        else:
            related_count_text = f"{related_count_text}건"  # 관련보도표시건수

        return f"""
                                    <span>　</span>
                                    <a href="{safe_url(related_url)}" target="_blank"
                                       style="font-size:11px; line-height:1.5;
                                              font-weight:800; color:#2563eb; text-decoration:none;">
                                        관련보도 {safe_text(related_count_text)} 보기
                                    </a>
        """

    # 2) 상세 페이지 URL이 없는 경우에만 메일 내부 fallback 목록을 만든다.
    #    Pages 설정이 없거나 관련 페이지 생성이 꺼져도, 상위 3건 정도는 메일에서 바로 확인할 수 있게 하기 위함이다.
    titles = news.get("group_article_titles") or []  # titles
    urls = news.get("group_article_urls") or []  # URL목록
    main_url = _normalize_url_for_compare(news.get("url"))  # mainURL

    related_items = []  # 관련보도항목목록
    seen_urls = set()  # 확인된URL목록
    seen_titles = set()  # 확인된titles

    # 3) group_article_titles와 group_article_urls는 같은 인덱스가 같은 기사라는 전제로 묶는다.
    #    URL/제목 중복을 같이 제거해 메일 내부 fallback 목록이 같은 기사로 반복되지 않게 한다.
    for title, url in zip(titles, urls):  # 제목,URL
        title_text = str(title or "").strip()  # 제목텍스트
        url_text = str(url or "").strip()  # URL텍스트
        normalized_url = _normalize_url_for_compare(url_text)  # 정규화URL
        normalized_title = re.sub(r"\s+", " ", title_text).strip().lower()  # 정규화제목

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
            "is_main": bool(main_url and normalized_url == main_url),
        })

    if not related_items:
        return ""

    # 4) 메일 내부에는 처음 3건만 보여주고 나머지는 "외 N건"으로 줄인다.
    #    메일 카드가 너무 길어지면 본문 요약보다 관련보도 목록이 더 눈에 띄기 때문이다.
    related_count = len(related_items)  # 관련보도건수
    visible_items = related_items[:3]  # 표시관련보도목록
    hidden_count = max(related_count - len(visible_items), 0)  # 숨김관련보도건수
    item_html = ""  # 항목HTML

    for index, item in enumerate(visible_items, 1):  # 순번,항목
        main_label = ""  # mainlabel
        if item.get("is_main"):
            main_label = (  # mainlabel
                """
                                                    <span style="font-size:10px; color:#737373; font-weight:700;">
                                                        대표
                                                    </span>
            """
            )

        item_html += f"""
                                        <tr>
                                            <td valign="top" width="22"
                                                style="padding:4px 0 4px 0; font-size:11px; line-height:1.45; color:#71717a;">
                                                {index}.
                                            </td>
                                            <td style="padding:4px 0 4px 0; font-size:12px; line-height:1.45;">
                                                <a href="{safe_url(item.get("url"))}" target="_blank"
                                                   style="font-weight:700; text-decoration:none; color:#1d4ed8;">
                                                    {safe_text(item.get("title"))}
                                                </a>
                                                {main_label}
                                            </td>
                                        </tr>
        """

    hidden_notice_html = ""  # hiddennoticeHTML
    if hidden_count:
        hidden_notice_html = f"""
                                        <tr>
                                            <td colspan="2"
                                                style="padding:5px 0 2px 22px; font-size:11px; line-height:1.45; color:#71717a;">
                                                외 {hidden_count}건 더 있음
                                            </td>
                                        </tr>
        """

    return f"""
                                <div style="margin:8px 0 0 0;">
                                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                                           style="border-collapse:collapse; margin:0;
                                                  background:#f8fafc; border:1px solid #e5e7eb;">
                                        <tr>
                                            <td style="padding:7px 9px 4px 9px;
                                                       font-size:11px; line-height:1.5;
                                                       font-weight:800; color:#2563eb;">
                                                관련보도 {related_count}건
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="padding:0 9px 7px 9px;">
                                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                                                       style="border-collapse:collapse;">
                                                    {item_html}
                                                    {hidden_notice_html}
                                                </table>
                                            </td>
                                        </tr>
                                    </table>
                                </div>
    """


# [코드 이해 주석]
# - 역할: 섹션 제목 아래, 핵심요약 3줄 위에 표시할 간단 대시보드.
# - 호출하는 곳: email_sender.build_news_section
# - 파라미터: section_result: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_section_dashboard(section_result):
    """
    섹션 제목 아래, 핵심요약 3줄 위에 표시할 간단 대시보드.

    표시 항목:
    - 전체검색수: 네이버 API 검색 결과로 확인한 전체 기사 수
    - 24시간초과제외: recent_hours 기준을 벗어나 제외된 기사 수
    - 반복이슈제외(3일): 최근 N일간 이미 발송된 이슈와 겹쳐 제외된 기사 수
    - 규칙기반제외: URL 중복, 제외 키워드, 그룹화 중복 대표화, 저품질/사진성 그룹 등
      AI 호출 전에 코드 규칙으로 제외한 기사 수
    - AI선별: 최종 중복 제거 후 실제 메일 요약 대상으로 남은 기사 수
    """
    if not section_result:
        return ""

    # section_result는 main.py가 섹션별로 만든 최종 결과이고,
    # scrape_stats에는 naver_news_scraper/main/news_grouper/news_selector/summarizer의 단계별 통계가 합쳐져 있다.
    summaries = section_result.get("summaries", []) or []                    # 섹션메일뉴스목록
    scrape_stats = section_result.get("scrape_stats", {}) or {}              # 섹션처리통계

    total_seen_count = safe_count(scrape_stats.get("total_seen_count", 0))    # 네이버응답확인건수
    old_news_count = safe_count(scrape_stats.get("old_news_count", 0))        # 시간초과제외건수

    issue_filter_days = safe_count(scrape_stats.get("issue_filter_days", 3), 3)  # 반복이슈비교기간일수
    issue_filter_excluded_count = safe_count(scrape_stats.get("issue_filter_excluded_count", 0))  # 반복이슈제외건수

    rule_based_excluded_count = safe_count(scrape_stats.get("rule_based_excluded_count", None), None)  # 규칙기반제외건수
    if rule_based_excluded_count is None:
        # 반복이슈제외는 별도 표시하므로 규칙기반제외에서 제외한다.
        # 규칙기반제외에는 URL 중복, 제외 키워드, 그룹화 중복 대표화, 저품질/사진성 제외만 포함한다.
        code_rule_excluded_count = safe_count(scrape_stats.get("code_rule_excluded_count", None), None)  # 기존호환용코드규칙제외수
        if code_rule_excluded_count is not None:
            rule_based_excluded_count = max(0, code_rule_excluded_count - issue_filter_excluded_count)  # 규칙기반제외건수
        else:
            rule_based_excluded_count = (  # 규칙기반제외건수
                safe_count(scrape_stats.get("duplicate_count", 0))
                + safe_count(scrape_stats.get("exclude_keyword_excluded_count", 0))
                + safe_count(scrape_stats.get("grouping_low_quality_article_count", 0))
                + safe_count(scrape_stats.get("grouping_duplicate_excluded_count", 0))
            )

    selected_count = safe_count(section_result.get("selected_count", len(summaries)))  # 최종메일선별뉴스수

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
                                    3일간 반복이슈제외
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {issue_filter_excluded_count}
                                </div>
                            </td>
                            <td width="20%" style="padding:8px 7px; border:1px solid #d4d4d4;">
                                <div style="font-size:11px; line-height:1.3; font-weight:800; color:#737373; margin:0 0 3px 0;">
                                    규칙기반제외
                                </div>
                                <div style="font-size:17px; line-height:1.25; font-weight:900;">
                                    {rule_based_excluded_count}
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



# [코드 이해 주석]
# - 역할: 메일 핵심 3줄 생성은 깊은 추론보다 짧은 종합이 중요하므로.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: model: str
# - 리턴값: dict 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _email_insight_reasoning_effort_kwargs(model: str) -> dict:
    """
    메일 핵심 3줄 생성은 깊은 추론보다 짧은 종합이 중요하므로
    GPT-5 계열에서는 reasoning_effort를 가장 낮게 시도한다.
    구버전 SDK가 이 인자를 직접 지원하지 않으면 호출 wrapper가 extra_body로 옮긴다.
    """
    if not is_gpt5_model(model):
        return {}

    effort = os.getenv("EMAIL_INSIGHT_REASONING_EFFORT", "minimal").strip().lower()  # effort
    if effort in {"", "none", "default", "off", "false", "0"}:
        return {}
    return {"reasoning_effort": effort}


# [코드 이해 주석]
# - 역할: OpenAI SDK/모델 호환성 방어용 호출 wrapper.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: **kwargs: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _create_chat_completion_for_email_insight(**kwargs):
    """
    OpenAI SDK/모델 호환성 방어용 호출 wrapper.
    신규 Chat Completions body 필드를 직접 받지 않는 SDK에서도 호출되게 한다.
    """
    return create_openai_chat_completion(client, logger, **kwargs)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: response: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _response_content(response) -> str:
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: response: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _response_finish_reason(response) -> str:
    try:
        return str(response.choices[0].finish_reason or "").strip()
    except Exception:
        return ""

# [코드 이해 주석]
# - 역할: 핵심 3줄 LLM 응답을 검증하고 정규화한다.
# - 호출하는 곳: email_sender.build_section_insights
# - 파라미터: insight_text: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_insight_lines(insight_text):
    """
    핵심 3줄 LLM 응답을 검증하고 정규화한다.
    '1. 2. 3.'처럼 번호만 있고 실제 문장이 없는 응답은 실패로 본다.
    """
    raw_lines = [str(line or "").strip() for line in str(insight_text or "").split("\n")]  # 원본lines
    cleaned = []  # cleaned

    for raw_line in raw_lines:  # 원본line
        if not raw_line:
            continue
        text = re.sub(r"^\s*[0-9]+[\.\)．、:]\s*", "", raw_line).strip()  # 텍스트
        text = re.sub(r"\s+", " ", text).strip(" -–—•·.")  # 텍스트

        # 번호/기호만 있는 줄은 제외한다.
        meaningful_chars = re.sub(r"[^0-9a-zA-Z가-힣]", "", text)  # 의미있는문자열
        if len(meaningful_chars) < 8:
            continue

        cleaned.append(text)
        if len(cleaned) >= 3:
            break

    if len(cleaned) != 3:
        return []

    return [f"{idx}. {text}" for idx, text in enumerate(cleaned, 1)]


# [코드 이해 주석]
# - 역할: 섹션별 핵심 3줄 생성.
# - 호출하는 곳: email_sender.build_news_section
# - 파라미터: section_title: Any, summaries: Any, scrape_stats: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
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

    use_ai_insight = safe_bool(                                             # 메일3줄AI요약사용여부
        scrape_stats.get("email_insight_ai") if isinstance(scrape_stats, dict) else None,
        EMAIL_INSIGHT_USE_AI_DEFAULT,
    )

    if not use_ai_insight:
        # AI 요약을 쓰지 않을 때는 중요도 높은 뉴스 요약 3개를 그대로 상단 흐름으로 사용한다.
        # 이 경로는 추가 OpenAI 호출이 없어서 섹션 수가 많아도 비용이 늘지 않는다.
        lines = []                                                          # 메일상단3줄목록
        sorted_summaries = sorted(                                          # 중요도순요약뉴스목록
            summaries,
            key=lambda news: safe_int(news.get("importance_score", 3)),
            reverse=True,  # reverse
        )
        for i, news in enumerate(sorted_summaries[:3], 1):  # i,뉴스
            summary = str(news.get("summary", "")).strip()                  # 뉴스요약문
            title = str(news.get("title", "")).strip()                      # 뉴스제목
            line_text = summary or title                                    # 상단라인후보문장
            if line_text:
                lines.append(f"{i}. {line_text}")

        if not lines:
            lines = ["1. 이 섹션의 핵심 요약을 생성하지 못했습니다."]  # lines
    else:
        # AI 요약을 쓸 때도 입력은 summaries에 있는 제목/요약/중요도만 사용한다.
        # 원문에 없는 흐름을 만들지 않도록 프롬프트에 제공된 사실만 종합하게 한다.
        news_text_list = []                                                 # AI입력뉴스블록목록

        for i, news in enumerate(summaries, 1):  # i,뉴스
            title = str(news.get("title", "")).strip()                      # 뉴스제목
            summary = str(news.get("summary", "")).strip()                  # 뉴스요약문
            source = str(news.get("source", "언론사 미상")).strip()         # 언론사명
            importance_score = str(news.get("importance_score", "3")).strip()  # 중요도점수

            news_text_list.append(
                f"{i}. [{source} / 중요도 {importance_score}] {title}\n요약: {summary}"
            )

        news_text = "\n\n".join(news_text_list)                             # AI입력뉴스목록텍스트

        prompt = f"""
"{section_title}" 섹션 뉴스만 보고 메일 상단 핵심 3줄을 작성하세요.

규칙:
1. 제공된 제목/요약에 있는 사실만 사용합니다.
2. 특정 기사 요약을 그대로 복사하지 말고, 선별된 뉴스 전체에서 반복되는 흐름과 핵심 변화를 종합합니다.
3. 뉴스 개수, 중요도 개수 같은 메타 설명은 쓰지 않습니다.
4. 서로 다른 3가지 관점으로 짧게 정리합니다.
5. 반드시 "1. ", "2. ", "3. "으로 시작하는 정확히 3줄만 출력합니다.

뉴스:
{news_text}
"""

        try:
            if client is None:
                raise ValueError("OPENAI_API_KEY가 없거나 OpenAI 클라이언트를 초기화할 수 없습니다.")

            messages = [                                                    # 메일3줄요약프롬프트메시지
                {
                    "role": "system",
                    "content": (
                        "뉴스 목록만 근거로 섹션의 핵심 흐름을 정확히 3줄로 요약합니다. "
                        "추측이나 원문에 없는 사실은 추가하지 않습니다. "
                        "반드시 본문을 출력해야 하며, 1~3번 번호 목록 외 텍스트는 쓰지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            attempt_limits = [                                              # AI응답토큰시도한도목록
                EMAIL_INSIGHT_MAX_COMPLETION_TOKENS if is_gpt5_model(MODEL) else min(300, EMAIL_INSIGHT_MAX_COMPLETION_TOKENS),
            ]
            if is_gpt5_model(MODEL):
                retry_limit = max(EMAIL_INSIGHT_RETRY_MAX_COMPLETION_TOKENS, EMAIL_INSIGHT_MAX_COMPLETION_TOKENS)  # 빈응답재시도토큰상한
                if retry_limit != attempt_limits[0]:
                    attempt_limits.append(retry_limit)

            insight_text = ""                                               # AI생성메일3줄원문
            total_insight_tokens = 0                                        # 메일3줄요약총토큰수

            for attempt_no, completion_limit in enumerate(attempt_limits, 1):  # 시도번호,응답상한
                response = _create_chat_completion_for_email_insight(       # 메일3줄요약AI응답
                    model=MODEL,  # 모델
                    messages=messages,  # messages
                    **openai_temperature_kwargs(MODEL, 0.2),
                    **openai_token_limit_kwargs(MODEL, completion_limit),
                    **_email_insight_reasoning_effort_kwargs(MODEL),
                )

                usage_info = record_openai_usage(  # usageinfo
                    logger,
                    f"[{section_title}] 메일 핵심 3줄 시도 {attempt_no}",
                    MODEL,
                    response.usage,
                )
                insight_tokens = usage_info["total_tokens"]                 # 이번시도토큰수
                total_insight_tokens += insight_tokens  # 처리값

                insight_text = _response_content(response)                  # 응답본문텍스트
                finish_reason = _response_finish_reason(response)           # 응답종료사유
                reasoning_tokens = safe_count(usage_info.get("reasoning_tokens", 0))  # 추론토큰수

                if insight_text:
                    break

                logger.warning(
                    f"⚠️ [{section_title}] 메일 핵심 3줄 응답 본문이 비어 있습니다. "
                    f"attempt={attempt_no}, finish_reason={finish_reason}, "
                    f"reasoning_tokens={reasoning_tokens}, completion_limit={completion_limit}"
                )

            if isinstance(scrape_stats, dict):
                # 메일 3줄 요약은 email_sender.py 안에서 추가로 발생하는 AI 호출이다.
                # main.py 최종 비용 로그가 이 호출까지 합산하도록 scrape_stats에 다시 누적한다.
                scrape_stats["insight_tokens"] = safe_count(scrape_stats.get("insight_tokens", 0)) + total_insight_tokens  # 처리값
            logger.info(f"🧾 [{section_title}] 메일 핵심 3줄 토큰 사용량: {total_insight_tokens}")

            lines = normalize_insight_lines(insight_text)  # lines

            if not lines:
                raise ValueError("핵심 3줄 응답이 비어 있거나 번호만 출력되었습니다.")

        except Exception as e:  # 예외객체
            logger.warning(f"⚠️ [{section_title}] 핵심 3줄 LLM 생성 실패로 기존 요약 기반 fallback을 사용합니다: {e}")

            lines = []  # lines
            for i, news in enumerate(summaries[:3], 1):  # i,뉴스
                summary = str(news.get("summary", "")).strip()  # 요약
                if summary:
                    lines.append(f"{i}. {summary}")

            if not lines:
                lines = ["1. 이 섹션의 핵심 요약을 생성하지 못했습니다."]  # lines

    html_lines = ""                                                       # 메일3줄HTML

    for line in lines[:3]:  # line
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


# [코드 이해 주석]
# - 역할: 섹션별 포인트 색상 반환.
# - 호출하는 곳: email_sender.build_news_section
# - 파라미터: index: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_section_color(index):
    """
    섹션별 포인트 색상 반환
    네이버메일 호환을 위해 class가 아니라 inline style에서 사용한다.
    """
    colors = [  # colors
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#7c3aed",
        "#ea580c"
    ]

    return colors[index % len(colors)]


# [코드 이해 주석]
# - 역할: 뉴스 섹션 HTML 생성.
# - 호출하는 곳: email_sender.create_html_email
# - 파라미터: section_result: Any, section_index: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_news_section(section_result, section_index):
    """
    뉴스 섹션 HTML 생성.

    구조:
    - 섹션 제목
    - 섹션별 간단 대시보드
    - 핵심요약 3줄
    - 뉴스 목록
    """
    section_title = section_result.get("section_name", f"뉴스 섹션 {section_index + 1}")  # 섹션제목
    summaries = section_result.get("summaries", []) or []                                 # 섹션뉴스요약목록
    section_color = get_section_color(section_index)                                      # 섹션포인트색상

    # 1) 섹션 컨테이너를 먼저 열고, 이후 대시보드/인사이트/뉴스 카드를 순서대로 이어 붙인다.
    #    네이버메일 호환성을 위해 div layout보다 table + inline style을 중심으로 만든다.
    # 아래 문자열은 메일 클라이언트 호환성을 위해 table 기반 HTML을 누적하는 시작점이다.
    html_body = f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="border-collapse:collapse; margin:0 0 22px 0;">
            <tr>
                <td style="border-left:4px solid {section_color}; padding:0 0 0 12px;">
                    <div style="font-size:17px; font-weight:900; line-height:1.35; color:{section_color}; margin:0 0 10px 0;">
                        {safe_text(section_title)}
                    </div>
    """

    # 2) scrape_stats 기반 운영 대시보드를 먼저 붙인다.
    #    사용자는 이 숫자로 "수집 → 제외 → AI선별" 과정에서 후보가 어떻게 줄었는지 메일 안에서 바로 볼 수 있다.
    html_body += build_section_dashboard(section_result)  # 처리값

    # 3) 섹션 핵심요약 3줄을 붙인다.
    #    email_insight_ai가 꺼져 있으면 AI 추가 호출 없이 상위 뉴스 요약에서 규칙 기반으로 만든다.
    html_body += build_section_insights(  # 처리값
        section_title=section_title,  # 섹션제목
        summaries=summaries,  # 요약목록
        scrape_stats=section_result.get("scrape_stats", {})
    )

    if not summaries:
        html_body += (  # 처리값
            """
                    <div style="font-size:13px; line-height:1.5; margin:0 0 12px 0;">
                        표시할 뉴스가 없습니다.
                    </div>
                </td>
            </tr>
        </table>
        """
        )
        return html_body

    # 4) summaries 하나가 메일의 뉴스 카드 하나가 된다.
    #    news dict는 summarizer가 만든 표준 결과이며, source/published_at/importance_score/group_* 메타를 함께 갖고 있다.
    for i, news in enumerate(summaries, 1):  # i,뉴스
        published_date = format_korean_datetime(                                          # 표시용발행일시
            news.get("published_at") or news.get("published_at_kst") or ""
        )

        title = safe_text(news.get("title", "제목 없음"))                                 # HTML이스케이프뉴스제목
        url = safe_url(news.get("url", "#"))                                               # 안전처리뉴스URL
        summary = news.get("summary", "")                                                  # 뉴스요약원문
        summary_html = safe_text(summary).replace("\n", "<br>")                            # HTML표시용요약

        source = safe_text(news.get("source", "언론사 미상") or "언론사 미상")              # 언론사명
        importance_score = safe_int(news.get("importance_score", 3))                       # 중요도점수
        # 5) 관련보도는 group_article_count가 2건 이상일 때만 표시한다.
        #    대표 기사 1건짜리 그룹에는 "관련보도 1건" 링크를 붙이지 않아 메타 줄을 짧게 유지한다.
        related_reports_html = ""                                                         # 관련보도링크HTML
        if safe_count(news.get("group_article_count"), 1) > 1:
            related_reports_html = build_related_reports_html(  # 관련보도reportsHTML
                news,
                toggle_id=f"related_{section_index + 1}_{i}",
            )

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
                                    <span style="font-weight:800;">{source}</span>
                                    <span>　</span>
                                    <span>{published_date}</span>
                                    <span>　</span>
                                    <span style="font-weight:800; color:#ea580c;">중요도 {importance_score}</span>
                                    {related_reports_html}
                                </div>

                                <div style="font-size:13px; line-height:1.65; margin:0; padding:0; word-break:keep-all;">
                                    {summary_html}
                                </div>
                            </td>
                        </tr>
                    </table>
        """

    html_body += (  # 처리값
        """
                </td>
            </tr>
        </table>
    """
    )

    return html_body


# [코드 이해 주석]
# - 역할: section_results 구조 정규화.
# - 호출하는 곳: email_sender.create_html_email, email_sender.send_email
# - 파라미터: section_results: Any = None, summaries: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
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


# [코드 이해 주석]
# - 역할: HTML 이메일 생성.
# - 호출하는 곳: email_sender.send_email
# - 파라미터: briefing_name: Any = None, subject_prefix: Any = None, section_results: Any = None, summaries: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def create_html_email(
    briefing_name=None,  # briefing이름
    subject_prefix=None,  # 메일제목prefix
    section_results=None,  # 섹션결과목록
    summaries=None  # 요약목록
):
    """
    HTML 이메일 생성
    네이버메일 호환을 위해 table layout + inline style 중심으로 작성한다.
    """
    if briefing_name is None:
        briefing_name = "뉴스 브리핑"  # briefing이름

    if subject_prefix is None:
        subject_prefix = briefing_name  # 메일제목prefix

    # 1) 예전 단일 summaries 입력과 현재 section_results 입력을 같은 구조로 맞춘다.
    #    이후 HTML 생성은 항상 section_results만 보도록 만들어 분기 복잡도를 줄인다.
    section_results = normalize_section_results(                                          # 표준섹션결과목록
        section_results=section_results,  # 섹션결과목록
        summaries=summaries  # 요약목록
    )

    today_text, now_kst = get_today_date_text()                                           # 메일생성일문구와현재시각
    mail_title = f"{today_text} - {subject_prefix}"                                       # 메일본문상단제목

    # 전체 HTML도 문자열 자체가 결과물이므로, 주석은 f-string 바깥에 둔다.
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
                        <td style="padding:12px 0 0 0;
                                   font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;">
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                                   style="border-collapse:collapse; margin:0 0 16px 0; pyborder:1px solid #e5e7eb;">
                                <tr>
                                    <td style="padding:12px 14px;">
                                        <div style="font-size:12px; line-height:1.65; margin:0; color:#52525b; word-break:keep-all;">
                                            본 메일은 AI를 통해 주요 뉴스를 자동으로 선별·요약하여 발송되는 뉴스 브리핑입니다.<br>
                                            향후 수신을 원치 않으시는 경우 이용 중인 메일 서비스에서 수신 차단을 설정하시거나, 본 메일에 회신해 주시면 감사하겠습니다.
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:0 0 12px 0;">
                            <div style="font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;
                                        font-size:22px; font-weight:900; line-height:1.35;
                                        letter-spacing:-0.5px;">
                                {safe_text(mail_title)}
                            </div>
                        </td>
                    </tr>

                    

                    <tr>
                        <td style="padding:0;
                                   font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;">
"""

    # 2) 섹션 순서대로 build_news_section()을 호출한다.
    #    main.py에서 만든 section_results 순서가 그대로 메일의 섹션 순서가 된다.
    for index, section_result in enumerate(section_results):  # 순번,섹션결과
        html_body += build_news_section(  # 처리값
            section_result=section_result,  # 섹션결과
            section_index=index  # 섹션순번
        )

    html_body += f"""
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:8px 0 0 0;
                                   font-family:'Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif;
                                   color:#a1a1aa; font-size:11px; line-height:1.5;">
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


# [코드 이해 주석]
# - 역할: 이메일로 뉴스 요약 발송.
# - 호출하는 곳: email_sender.send_test_email, main.main
# - 파라미터: summaries: Any = None, subject: Any = None, briefing_name: Any = None, subject_prefix: Any = None,
# section_results: Any = None, receiver_env_name: Any = None, send_mode: Any = 'individual'
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 수신자와 메시지를 준비합니다 -> SMTP로 전송합니다 -> 성공 여부와 로그를 남깁니다.
def send_email(
    summaries=None,  # 요약목록
    subject=None,  # 메일제목
    briefing_name=None,  # briefing이름
    subject_prefix=None,  # 메일제목prefix
    section_results=None,  # 섹션결과목록
    receiver_env_name=None,  # 수신자env이름
    send_mode="individual"
):
    """
    이메일로 뉴스 요약 발송

    수신자 형식:
    EMAIL_RECEIVER=김선빈|seonbin.kim@lotte.net,고우탁|wootak.ko@lotte.net
    """
    try:
        # 1) 발송에 필요한 환경변수를 먼저 확인한다.
        #    SMTP 계정 정보가 없으면 HTML 생성이나 수신자 파싱을 진행해도 실제 발송은 불가능하다.
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

        # 2) 메일 본문에 들어갈 뉴스 구조를 표준 section_results로 맞춘다.
        #    main.py는 여러 섹션을 넘기고, 테스트/이전 호출부는 summaries만 넘길 수 있어 여기서 호환한다.
        section_results = normalize_section_results(                                      # 표준섹션결과목록
            section_results=section_results,  # 섹션결과목록
            summaries=summaries  # 요약목록
        )

        if briefing_name is None:
            briefing_name = "뉴스 브리핑"  # briefing이름

        if subject_prefix is None:
            subject_prefix = briefing_name  # 메일제목prefix

        selected_receiver_env_name = determine_receiver_env_name(                         # 실제사용수신자환경변수명
            briefing_name=briefing_name,  # briefing이름
            subject_prefix=subject_prefix,  # 메일제목prefix
            section_results=section_results,  # 섹션결과목록
            receiver_env_name=receiver_env_name  # 수신자env이름
        )

        # 3) receiver_env_name이 가리키는 환경변수에서 수신자 목록을 읽는다.
        #    "이름|메일" 또는 "메일" 형식을 normalize_receiver_info()가 최종 display 주소로 바꾼다.
        receiver_list = get_receiver_list(selected_receiver_env_name)                     # 발송대상수신자목록
        selected_send_mode = normalize_email_send_mode(send_mode)                         # 정규화발송방식

        if not receiver_list:
            logger.error(f"❌ 수신자 이메일이 설정되지 않았습니다: {selected_receiver_env_name}")
            return {
                "success": False,
                "message": f"{selected_receiver_env_name} 환경변수를 확인하세요"
            }

        # 4) 제목과 HTML 본문을 만든다.
        #    HTML은 모든 수신자에게 동일하지만, individual 발송에서는 To 헤더만 수신자별로 달라진다.
        if subject is None:
            today_text, _ = get_today_date_text()                                        # 메일제목날짜문구
            subject = f"📰 {today_text} - {subject_prefix}"                              # 최종메일제목

        html_body = create_html_email(                                                    # 발송HTML본문
            briefing_name=briefing_name,  # briefing이름
            subject_prefix=subject_prefix,  # 메일제목prefix
            section_results=section_results  # 섹션결과목록
        )

        success_count = 0                                                                 # 발송성공수
        failed_list = []                                                                  # 발송실패수신자목록

        logger.info(
            "📧 이메일 발송 시작: 수신자 %s명 / env=%s / mode=%s",
            len(receiver_list),
            selected_receiver_env_name,
            selected_send_mode,
        )

        # 5) SMTP 연결은 한 번만 열고, 발송 방식에 따라 bulk 또는 individual로 보낸다.
        #    bulk는 한 통에 여러 수신자를 넣고, individual은 같은 본문을 수신자별로 따로 보낸다.
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:  # 서버
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)

            if selected_send_mode == "bulk":
                normalized_receivers = []                                                 # 전체발송정규화수신자목록

                for receiver_info in receiver_list:  # 수신자info
                    try:
                        normalized_receivers.append(normalize_receiver_info(receiver_info))
                    except Exception as e:  # 예외객체
                        display_name = str(receiver_info)  # 표시건수이름
                        logger.error(f"   ❌ {display_name} 수신자 정보 오류: {str(e)}")
                        failed_list.append(display_name)

                if normalized_receivers:
                    receiver_emails = [receiver["email"] for receiver in normalized_receivers]  # 전체발송이메일주소목록
                    to_header = ", ".join(receiver["display"] for receiver in normalized_receivers)  # 전체발송To헤더

                    msg = build_email_message(                                            # 전체발송MIME메시지
                        subject=subject,  # 메일제목
                        html_body=html_body,  # HTML본문
                        to_header=to_header,  # toheader
                    )

                    server.send_message(
                        msg,
                        from_addr=EMAIL_SENDER,  # 환경값addr
                        to_addrs=receiver_emails,  # toaddrs
                    )
                    success_count = len(normalized_receivers)  # success건수
                    logger.debug("전체 이메일 발송 완료: %s", ", ".join(receiver_emails))

            else:
                for receiver_info in receiver_list:  # 수신자info
                    receiver_name = ""                                                    # 개별수신자이름
                    receiver_email = ""                                                   # 개별수신자이메일

                    try:
                        normalized_receiver = normalize_receiver_info(receiver_info)       # 정규화수신자정보
                        receiver_name = normalized_receiver["name"]                        # 수신자이름
                        receiver_email = normalized_receiver["email"]                      # 수신자이메일
                        receiver_display = normalized_receiver["display"]                  # To헤더표시값

                        msg = build_email_message(                                        # 개별발송MIME메시지
                            subject=subject,  # 메일제목
                            html_body=html_body,  # HTML본문
                            to_header=receiver_display,  # toheader
                        )

                        server.send_message(msg)
                        logger.debug(f"이메일 발송 완료: {receiver_name} ({receiver_email})")
                        success_count += 1  # 처리값

                    except Exception as e:  # 예외객체
                        display_name = receiver_name or receiver_email or str(receiver_info)  # 표시건수이름
                        logger.error(f"   ❌ {display_name} 발송 실패: {str(e)}")
                        failed_list.append(display_name)

        # 6) 성공/부분성공/실패를 호출자(main.py)가 로그와 히스토리 저장 여부에 사용할 수 있게 dict로 정리한다.
        if selected_send_mode == "bulk" and success_count == len(receiver_list):
            logger.info(f"🎉 전체 이메일 발송 완료! ({success_count}명)")
            return {
                "success": True,
                "message": f"{success_count}명에게 전체 발송 완료"
            }

        if selected_send_mode == "bulk" and success_count > 0:
            logger.warning(f"⚠️ 전체 발송 일부 완료: {success_count}/{len(receiver_list)}명")
            return {
                "success": True,
                "message": f"전체 발송 {success_count}명 성공, {len(failed_list)}명 실패: {', '.join(failed_list)}"
            }

        if selected_send_mode == "bulk":
            logger.error("❌ 전체 이메일 발송 실패")
            return {
                "success": False,
                "message": f"전체 발송 실패: {', '.join(failed_list)}"
            }

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

    except Exception as e:  # 예외객체
        logger.error(f"❌ 이메일 발송 실패: {str(e)}")
        return {
            "success": False,
            "message": f"이메일 발송 실패: {str(e)}"
        }


# [코드 이해 주석]
# - 역할: 이메일 전송만 테스트한다.
# - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
# - 파라미터: receiver_env_name: Any = 'EMAIL_RECEIVER'
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 수신자와 메시지를 준비합니다 -> SMTP로 전송합니다 -> 성공 여부와 로그를 남깁니다.
def send_test_email(receiver_env_name="EMAIL_RECEIVER"):
    """
    이메일 전송만 테스트한다.

    - 뉴스 수집 안 함
    - 뉴스 요약 안 함
    - OpenAI 호출 안 함
    - SMTP 로그인/전송/수신자 파싱만 확인
    - 섹션별 대시보드 위치 확인 가능
    """
    today_text, _ = get_today_date_text()  # today텍스트,_

    test_section_results = [  # 테스트섹션결과목록
        {
            "section_name": "경제 뉴스 브리핑",
            "summaries": [
                {
                    "title": "테스트 경제 뉴스 제목입니다",
                    "summary": "경제 뉴스 요약 테스트 문장입니다.",
                    "url": "#",
                    "published_at": "",
                    "importance_score": 4,
                    "source": "테스트언론",
                    "group_article_count": 8,
                    "group_source_count": 5
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
                    "source": "테스트언론",
                    "group_article_count": 3,
                    "group_source_count": 2
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
        section_results=test_section_results,  # 섹션결과목록
        receiver_env_name=receiver_env_name  # 수신자env이름
    )


if __name__ == "__main__":
    result = send_test_email(receiver_env_name="EMAIL_RECEIVER")  # 결과
    logger.info(result)
