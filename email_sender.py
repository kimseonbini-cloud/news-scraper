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
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

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

        kst = pytz.timezone('Asia/Seoul')
        dt_kst = dt.astimezone(kst)

        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        weekday = weekdays[dt_kst.weekday()]

        hour = dt_kst.hour
        if hour < 12:
            period = '오전'
            display_hour = hour if hour > 0 else 12
        else:
            period = '오후'
            display_hour = hour if hour == 12 else hour - 12

        return f"{dt_kst.year}년 {dt_kst.month:02d}월 {dt_kst.day:02d}일 {weekday} {period} {display_hour:02d}:{dt_kst.minute:02d}"

    except Exception:
        return safe_text(date_string)


def get_today_title():
    """
    메일 제목용 날짜 생성
    예: 2026년 05월 08일 금요일 - 뉴스 브리핑
    """
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)

    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    weekday = weekdays[now_kst.weekday()]

    formatted_date = f"{now_kst.year}년 {now_kst.month:02d}월 {now_kst.day:02d}일 {weekday}"

    return f"{formatted_date} - 뉴스 브리핑", now_kst


def build_section_insights(section_title, summaries):
    """
    섹션별 핵심 3줄 생성

    의료 뉴스는 의료 뉴스끼리,
    롯데 뉴스는 롯데 뉴스끼리 따로 요약한다.
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
        html_body += f"""
                <div class="empty-section">
                    표시할 뉴스가 없습니다.
                </div>
            </div>
        """
        return html_body

    for i, news in enumerate(summaries, 1):
        published_date = format_korean_datetime(news.get('published_at', ''))

        title = safe_text(news.get('title', '제목 없음'))
        url = safe_url(news.get('url', '#'))
        summary = news.get('summary', '')
        summary_html = safe_text(summary).replace('\n', '<br>')

        category = safe_text(news.get('category', default_keyword) or default_keyword)
        importance_score = safe_int(news.get('importance_score', 3))
        stars = '★' * importance_score + '☆' * (5 - importance_score)

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


def create_html_email(medical_summaries=None, lotte_summaries=None, summaries=None):
    """
    HTML 이메일 생성

    Args:
        medical_summaries: 의료 뉴스 요약 리스트
        lotte_summaries: 롯데 관련 뉴스 요약 리스트
        summaries: 기존 호환용 뉴스 요약 리스트

    Returns:
        str: HTML 형식의 이메일 본문
    """
    if medical_summaries is None:
        medical_summaries = []

    if lotte_summaries is None:
        lotte_summaries = []

    # 기존 코드 호환용
    # 예전처럼 create_html_email(summaries)로 호출된 경우도 의료 뉴스로 처리
    if summaries is not None and not medical_summaries:
        medical_summaries = summaries

    mail_title, now_kst = get_today_title()

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
                padding: 0 0 16px 0;
                border-bottom: 1px solid #dddddd;
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

            .medical-section {{
                border-left: 3px solid #2563eb;
                padding-left: 10px;
            }}

            .medical-section .section-title {{
                color: #2563eb;
            }}

            .lotte-section {{
                border-left: 3px solid #dc2626;
                padding-left: 10px;
            }}

            .lotte-section .section-title {{
                color: #dc2626;
            }}

            .section-insights {{
                margin: 0 0 14px 0;
                padding: 0 0 10px 0;
                border-bottom: 1px solid #eeeeee;
                overflow: visible;
            }}

            .section-insights-title {{
                margin: 0 0 6px 0;
                padding: 0;
                color: #555555;
                font-size: 12px;
                font-weight: 900;
                line-height: 1.3;
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

            .footer-icon {{
                display: none;
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

                .section-insights-title {{
                    color: #cccccc !important;
                }}

                .insight-line {{
                    color: #eeeeee !important;
                }}

                .medical-section {{
                    border-left-color: #93c5fd !important;
                }}

                .medical-section .section-title {{
                    color: #93c5fd !important;
                }}

                .lotte-section {{
                    border-left-color: #fca5a5 !important;
                }}

                .lotte-section .section-title {{
                    color: #fca5a5 !important;
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

    html_body += build_news_section(
        section_title="의료 뉴스 브리핑",
        summaries=medical_summaries,
        default_keyword="의료뉴스",
        section_class="medical-section"
    )

    html_body += build_news_section(
        section_title="롯데 뉴스 브리핑",
        summaries=lotte_summaries,
        default_keyword="롯데",
        section_class="lotte-section"
    )

    html_body += f"""
            </div>

            <div class="footer">
                <p><strong>이 메일은 AI를 통하여 자동으로 선별, 요약되어 발송되었습니다.</strong></p>
                <p>EMR & Lotte News Scraper v1.0</p>
                <p style="color: #adb5bd; margin-top: 4px;">
                    © {now_kst.year} EMR & Lotte News Scraper. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_body


def get_receiver_list():
    """
    환경변수에서 수신자 목록 가져오기
    """
    if EMAIL_RECEIVER:
        receivers = [email.strip() for email in EMAIL_RECEIVER.split(',')]
        receivers = [email for email in receivers if email]
        return receivers
    return []


def send_email(summaries=None, subject=None, medical_summaries=None, lotte_summaries=None):
    """
    이메일로 뉴스 요약 발송

    Args:
        summaries: 기존 호환용 뉴스 요약 리스트
        subject: 이메일 제목
        medical_summaries: 의료 뉴스 요약 리스트
        lotte_summaries: 롯데 관련 뉴스 요약 리스트

    Returns:
        dict: 발송 결과
    """
    try:
        receiver_list = get_receiver_list()

        if not receiver_list:
            logger.error("❌ 수신자 이메일이 설정되지 않았습니다.")
            return {
                'success': False,
                'message': 'EMAIL_RECEIVER 환경변수를 확인하세요'
            }

        if medical_summaries is None:
            medical_summaries = []

        if lotte_summaries is None:
            lotte_summaries = []

        # 기존 send_email(summaries) 호출 호환
        if summaries is not None and not medical_summaries:
            medical_summaries = summaries

        if subject is None:
            subject, _ = get_today_title()
            subject = f"📰 {subject}"

        html_body = create_html_email(
            medical_summaries=medical_summaries,
            lotte_summaries=lotte_summaries
        )

        success_count = 0
        failed_list = []

        logger.info(f"📧 이메일 개별 발송 시작: {len(receiver_list)}명")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)

            for receiver in receiver_list:
                try:
                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = subject
                    msg['From'] = EMAIL_SENDER
                    msg['To'] = receiver

                    html_part = MIMEText(html_body, 'html', 'utf-8')
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
                'success': True,
                'message': f'{success_count}명에게 개별 발송 완료'
            }

        elif success_count > 0:
            logger.warning(f"⚠️ 일부 발송 완료: {success_count}/{len(receiver_list)}명")
            return {
                'success': True,
                'message': f'{success_count}명 성공, {len(failed_list)}명 실패: {", ".join(failed_list)}'
            }

        else:
            logger.error("❌ 모든 이메일 발송 실패")
            return {
                'success': False,
                'message': '모든 발송 실패'
            }

    except smtplib.SMTPAuthenticationError:
        logger.error("❌ 로그인 실패! 이메일/비밀번호를 확인하세요.")
        return {
            'success': False,
            'message': 'SMTP 인증 실패 (앱 비밀번호 확인 필요)'
        }

    except Exception as e:
        logger.error(f"❌ 이메일 발송 실패: {str(e)}")
        return {
            'success': False,
            'message': f'이메일 발송 실패: {str(e)}'
        }