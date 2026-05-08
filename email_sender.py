"""
이메일 자동 발송 모듈
"""
import smtplib
import os
import html as html_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
import logging
import pytz
from email.utils import parsedate_to_datetime
from openai import OpenAI

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

# 개인 경제 브리핑용 수신자
EMAIL_RECEIVER_ECONOMY = os.getenv("EMAIL_RECEIVER_ECONOMY")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ====================================
# OpenAI 설정
# ====================================
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def format_korean_datetime(date_string):
    """
    날짜 문자열을 한국식 표현으로 변환
    """
    try:
        dt = parsedate_to_datetime(date_string)

        kst = pytz.timezone("Asia/Seoul")
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

        return f"{dt_kst.year}년 {dt_kst.month:02d}월 {dt_kst.day:02d}일 {weekday} {period} {display_hour:02d}:{dt_kst.minute:02d}"

    except Exception:
        return safe_text(date_string)


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


def determine_receiver_env_name(briefing_name=None, subject_prefix=None, section_results=None, receiver_env_name=None):
    """
    브리핑 종류에 따라 사용할 수신자 환경변수명 결정

    우선순위:
    1. send_email에서 receiver_env_name을 직접 넘긴 경우
    2. briefing_name / subject_prefix / section_name에 경제·부동산·증권 키워드가 있으면 EMAIL_RECEIVER_ECONOMY
    3. 기본 EMAIL_RECEIVER
    """
    if receiver_env_name:
        return receiver_env_name

    text_parts = [
        str(briefing_name or ""),
        str(subject_prefix or "")
    ]

    if section_results:
        for section in section_results:
            text_parts.append(str(section.get("section_name", "")))

    combined_text = " ".join(text_parts)

    economy_keywords = ["경제", "부동산", "증권", "코스피", "코스닥", "환율", "금리"]

    if any(keyword in combined_text for keyword in economy_keywords):
        return "EMAIL_RECEIVER_ECONOMY"

    return "EMAIL_RECEIVER"


def get_receiver_list(receiver_env_name=None):
    """
    환경변수에서 수신자 목록 가져오기

    Args:
        receiver_env_name:
            EMAIL_RECEIVER
            EMAIL_RECEIVER_ECONOMY
            등 환경변수명
    """
    if receiver_env_name is None:
        receiver_env_name = "EMAIL_RECEIVER"

    receiver_value = os.getenv(receiver_env_name)

    if receiver_value:
        receivers = [email.strip() for email in receiver_value.split(",")]
        receivers = [email for email in receivers if email]
        return receivers

    return []


def build_section_insights(section_title, summaries):
    """
    섹션별 핵심 3줄 생성
    """
    if not summaries:
        return """
                <div class="section-insights">
                    <div class="insight-line">1. 이 섹션에서 요약할 뉴스가 없습니다.</div>
                </div>
        """

    news_text_list = []

    for i, news in enumerate(summaries, 1):
        title = str(news.get("title", "")).strip()
        summary = str(news.get("summary", "")).strip()
        category = str(news.get("category", "기타")).strip()
        importance_score = str(news.get("importance_score", "3")).strip()

        news_text_list.append(
            f"{i}. [{category} / 중요도 {importance_score}] {title}\n요약: {summary}"
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
        lines = [line.strip() for line in insight_text.split("\n") if line.strip()]
        lines = lines[:3]

        if not lines:
            raise ValueError("핵심 3줄 응답이 비어 있습니다.")

        html_lines = ""
        for line in lines:
            html_lines += f"""
                    <div class="insight-line">{safe_text(line)}</div>
            """

        return f"""
                <div class="section-insights">
                    {html_lines}
                </div>
        """

    except Exception as e:
        logger.error(f"❌ [{section_title}] 핵심 3줄 생성 실패: {e}")

        fallback_lines = []

        for i, news in enumerate(summaries[:3], 1):
            summary = str(news.get("summary", "")).strip()
            if summary:
                fallback_lines.append(f"{i}. {summary}")

        if not fallback_lines:
            fallback_lines = ["1. 이 섹션의 핵심 요약을 생성하지 못했습니다."]

        html_lines = ""
        for line in fallback_lines[:3]:
            html_lines += f"""
                    <div class="insight-line">{safe_text(line)}</div>
            """

        return f"""
                <div class="section-insights">
                    {html_lines}
                </div>
        """


def get_section_class(index):
    """
    섹션별 색상 클래스 순환
    """
    classes = [
        "section-blue",
        "section-red",
        "section-green",
        "section-purple",
        "section-orange"
    ]

    return classes[index % len(classes)]


def build_news_section(section_title, summaries, default_keyword, section_class):
    """
    뉴스 섹션 HTML 생성
    """
    html_body = f"""
            <div class="news-section {section_class}">
                <div class="section-title">{safe_text(section_title)}</div>
    """

    html_body += build_section_insights(
        section_title=section_title,
        summaries=summaries
    )

    if not summaries:
        html_body += """
                <div class="empty-section">
                    표시할 뉴스가 없습니다.
                </div>
            </div>
        """
        return html_body

    for i, news in enumerate(summaries, 1):
        published_date = format_korean_datetime(news.get("published_at", ""))

        title = safe_text(news.get("title", "제목 없음"))
        url = safe_url(news.get("url", "#"))
        summary = news.get("summary", "")
        summary_html = safe_text(summary).replace("\n", "<br>")

        category = safe_text(news.get("category", default_keyword) or default_keyword)
        importance_score = safe_int(news.get("importance_score", 3))
        stars = "★" * importance_score + "☆" * (5 - importance_score)

        html_body += f"""
                <!-- {safe_text(section_title)} 뉴스 {i} -->
                <div class="news-item">
                    <div class="news-header">
                        <h2 class="news-title">
                            <a href="{url}" target="_blank">{i}. {title}</a>
                        </h2>
                    </div>

                    <div class="news-meta">
                        <span class="category-badge">{category}</span>
                        <span class="importance-badge">중요도 {importance_score}</span>
                        <span class="importance-stars">{stars}</span>
                        <span class="publish-date">{published_date}</span>
                    </div>

                    <div class="news-summary">
                        {summary_html}
                    </div>
                </div>
        """

    html_body += """
            </div>
    """

    return html_body


def normalize_section_results(section_results=None, summaries=None):
    """
    section_results 구조 정규화

    원칙:
    - 섹션명은 config의 section_name을 사용한다.
    - 의료/롯데 같은 이름을 하드코딩하지 않는다.
    - 구버전 단일 summaries 호출만 최소한으로 호환한다.
    """
    if section_results is not None:
        return section_results

    if summaries is not None:
        return [
            {
                "section_name": "뉴스 브리핑",
                "summaries": summaries
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

    신규 방식:
        create_html_email(
            briefing_name="경제 뉴스 브리핑",
            subject_prefix="경제·부동산·증권 뉴스 브리핑",
            section_results=[
                {"section_name": "경제 뉴스 브리핑", "summaries": [...]},
                {"section_name": "부동산 뉴스 브리핑", "summaries": [...]}
            ]
        )

    기존 방식도 임시 호환:
        create_html_email(medical_summaries=[...], lotte_summaries=[...])
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
        <style>
            :root {{
                color-scheme: light dark;
                supported-color-schemes: light dark;
            }}

            body {{
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', Arial, sans-serif;
                margin: 0;
                padding: 12px;
                color: #222222;
                font-size: 13px;
                line-height: 1.45;
                text-align: left;
                -webkit-text-size-adjust: 100%;
            }}

            .container {{
                width: 100%;
                max-width: 860px;
                margin: 0;
                padding: 0;
                border: 0;
                box-shadow: none;
                overflow: visible;
            }}

            .header {{
                margin: 0 0 14px 0;
                padding: 0 0 10px 0;
                border-bottom: 2px solid #222222;
                color: #111111;
            }}

            .header h1 {{
                margin: 0;
                padding: 0;
                font-size: 18px;
                font-weight: 900;
                line-height: 1.3;
                letter-spacing: -0.4px;
                color: #111111;
            }}

            .content {{
                margin: 0;
                padding: 0;
            }}

            .news-section {{
                margin: 0 0 20px 0;
                padding: 0 0 16px 10px;
                border-bottom: 1px solid #dddddd;
                border-left: 3px solid #666666;
                overflow: visible;
            }}

            .section-title {{
                margin: 0 0 8px 0;
                padding: 0;
                font-size: 16px;
                font-weight: 900;
                line-height: 1.3;
                letter-spacing: -0.2px;
            }}

            .section-blue {{
                border-left-color: #2563eb;
            }}

            .section-blue .section-title {{
                color: #2563eb;
            }}

            .section-red {{
                border-left-color: #dc2626;
            }}

            .section-red .section-title {{
                color: #dc2626;
            }}

            .section-green {{
                border-left-color: #16a34a;
            }}

            .section-green .section-title {{
                color: #16a34a;
            }}

            .section-purple {{
                border-left-color: #7c3aed;
            }}

            .section-purple .section-title {{
                color: #7c3aed;
            }}

            .section-orange {{
                border-left-color: #ea580c;
            }}

            .section-orange .section-title {{
                color: #ea580c;
            }}

            .section-insights {{
                margin: 0 0 14px 0;
                padding: 0 0 10px 0;
                border-bottom: 1px solid #eeeeee;
                overflow: visible;
            }}

            .insight-line {{
                margin: 3px 0;
                padding: 0;
                color: #333333;
                font-size: 12px;
                line-height: 1.5;
                word-break: keep-all;
                overflow-wrap: anywhere;
            }}

            .news-item {{
                margin: 0 0 14px 0;
                padding: 0 0 12px 0;
                border: 0;
                border-bottom: 1px solid #eeeeee;
                box-shadow: none;
                overflow: visible;
            }}

            .news-item:last-child {{
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: 0;
            }}

            .news-header {{
                display: block;
                margin: 0 0 4px 0;
                padding: 0;
            }}

            .news-title {{
                margin: 0;
                padding: 0;
                font-size: 14px;
                font-weight: 800;
                line-height: 1.38;
                letter-spacing: -0.25px;
                color: #111111;
                word-break: keep-all;
                overflow: visible;
                height: auto;
                max-height: none;
            }}

            .news-title a {{
                color: #111111;
                text-decoration: none;
            }}

            .news-title a:hover {{
                color: #2563eb;
                text-decoration: underline;
            }}

            .news-meta {{
                display: block;
                margin: 4px 0 6px 0;
                padding: 0;
                line-height: 1.35;
            }}

            .category-badge {{
                display: inline-block;
                margin: 0 6px 3px 0;
                padding: 0;
                color: #555555;
                font-size: 10px;
                font-weight: 800;
                line-height: 1.4;
            }}

            .importance-badge {{
                display: inline-block;
                margin: 0 6px 3px 0;
                padding: 0;
                color: #c2410c;
                font-size: 10px;
                font-weight: 800;
                line-height: 1.4;
            }}

            .importance-stars {{
                display: inline-block;
                margin: 0 6px 3px 0;
                color: #d97706;
                font-size: 10px;
                letter-spacing: -1px;
                line-height: 1.4;
            }}

            .publish-date {{
                display: inline-block;
                margin: 0 0 3px 0;
                color: #777777;
                font-size: 10px;
                font-weight: 500;
                line-height: 1.4;
            }}

            .news-summary {{
                display: block;
                margin: 0;
                padding: 0;
                color: #333333;
                border: 0;
                border-radius: 0;
                font-size: 12px;
                line-height: 1.55;
                letter-spacing: -0.15px;
                word-break: keep-all;
                overflow-wrap: anywhere;
                white-space: normal;
                overflow: visible;
                height: auto;
                max-height: none;
                -webkit-line-clamp: unset;
                -webkit-box-orient: unset;
            }}

            .empty-section {{
                margin: 0;
                padding: 4px 0 2px 0;
                color: #777777;
                font-size: 12px;
                line-height: 1.4;
            }}

            .footer {{
                margin: 4px 0 0 0;
                padding: 8px 2px 0 2px;
                color: #999999;
                border: 0;
                font-size: 10px;
                line-height: 1.35;
                text-align: left;
            }}

            .footer p {{
                margin: 2px 0;
                padding: 0;
            }}

            @media (prefers-color-scheme: dark) {{
                body {{
                    color: #eeeeee !important;
                }}

                .header {{
                    border-bottom-color: #eeeeee !important;
                    color: #ffffff !important;
                }}

                .header h1 {{
                    color: #ffffff !important;
                }}

                .news-section {{
                    border-bottom-color: #555555 !important;
                }}

                .section-insights {{
                    border-bottom-color: #444444 !important;
                }}

                .insight-line {{
                    color: #eeeeee !important;
                }}

                .section-blue {{
                    border-left-color: #93c5fd !important;
                }}

                .section-blue .section-title {{
                    color: #93c5fd !important;
                }}

                .section-red {{
                    border-left-color: #fca5a5 !important;
                }}

                .section-red .section-title {{
                    color: #fca5a5 !important;
                }}

                .section-green {{
                    border-left-color: #86efac !important;
                }}

                .section-green .section-title {{
                    color: #86efac !important;
                }}

                .section-purple {{
                    border-left-color: #c4b5fd !important;
                }}

                .section-purple .section-title {{
                    color: #c4b5fd !important;
                }}

                .section-orange {{
                    border-left-color: #fdba74 !important;
                }}

                .section-orange .section-title {{
                    color: #fdba74 !important;
                }}

                .news-item {{
                    border-bottom-color: #444444 !important;
                }}

                .news-title,
                .news-title a {{
                    color: #ffffff !important;
                }}

                .news-title a:hover {{
                    color: #93c5fd !important;
                }}

                .category-badge {{
                    color: #dddddd !important;
                }}

                .importance-badge {{
                    color: #fdba74 !important;
                }}

                .importance-stars {{
                    color: #fbbf24 !important;
                }}

                .publish-date {{
                    color: #bbbbbb !important;
                }}

                .news-summary {{
                    color: #eeeeee !important;
                }}

                .empty-section {{
                    color: #bbbbbb !important;
                }}

                .footer {{
                    color: #bbbbbb !important;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{safe_text(mail_title)}</h1>
            </div>

            <div class="content">
    """

    for index, section_result in enumerate(section_results):
        section_name = section_result.get("section_name", f"뉴스 섹션 {index + 1}")
        summaries_for_section = section_result.get("summaries", [])
        section_class = get_section_class(index)

        html_body += build_news_section(
            section_title=section_name,
            summaries=summaries_for_section,
            default_keyword=section_name,
            section_class=section_class
        )

    html_body += f"""
            </div>

            <div class="footer">
                <p><strong>이 메일은 AI를 통하여 자동으로 선별, 요약되어 발송되었습니다.</strong></p>
                <p>{safe_text(briefing_name)} v1.0</p>
                <p style="color: #adb5bd; margin-top: 4px;">
                    © {now_kst.year} {safe_text(briefing_name)}. All rights reserved.
                </p>
            </div>
        </div>
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

    신규 방식:
        send_email(
            briefing_name=briefing_name,
            subject_prefix=subject_prefix,
            section_results=section_results
        )

    수신자 분리:
        사내용: EMAIL_RECEIVER
        경제용: EMAIL_RECEIVER_ECONOMY
    """
    try:
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

            for receiver in receiver_list:
                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"] = EMAIL_SENDER
                    msg["To"] = receiver

                    html_part = MIMEText(html_body, "html", "utf-8")
                    msg.attach(html_part)

                    server.send_message(msg)
                    logger.info(f"   ✅ {receiver} 발송 완료")
                    success_count += 1

                except Exception as e:
                    logger.error(f"   ❌ {receiver} 발송 실패: {str(e)}")
                    failed_list.append(receiver)

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
            "message": "모든 발송 실패"
        }

    except smtplib.SMTPAuthenticationError:
        logger.error("❌ 로그인 실패! 이메일/비밀번호를 확인하세요.")
        return {
            "success": False,
            "message": "SMTP 인증 실패. 앱 비밀번호 확인 필요"
        }

    except Exception as e:
        logger.error(f"❌ 이메일 발송 실패: {str(e)}")
        return {
            "success": False,
            "message": f"이메일 발송 실패: {str(e)}"
        }