"""
이메일 자동 발송 모듈
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
import logging
import pytz
from email.utils import parsedate_to_datetime

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


def format_korean_datetime(date_string):
    """
    날짜 문자열을 한국식 표현으로 변환
    
    Args:
        date_string: 원본 날짜 문자열 (예: "Thu, 07 May 2026 12:06:00 +0900")
    
    Returns:
        str: 한국식 날짜 (예: "2026년 05월 07일 목요일 오후 12:06")
    """
    try:
        # RFC 2822 형식 파싱
        dt = parsedate_to_datetime(date_string)
        
        # 한국 시간대로 변환
        kst = pytz.timezone('Asia/Seoul')
        dt_kst = dt.astimezone(kst)
        
        # 요일 매핑
        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        weekday = weekdays[dt_kst.weekday()]
        
        # 오전/오후 구분
        hour = dt_kst.hour
        if hour < 12:
            period = '오전'
            display_hour = hour if hour > 0 else 12
        else:
            period = '오후'
            display_hour = hour if hour == 12 else hour - 12
        
        # 한국식 형식으로 변환
        return f"{dt_kst.year}년 {dt_kst.month:02d}월 {dt_kst.day:02d}일 {weekday} {period} {display_hour:02d}:{dt_kst.minute:02d}"
    
    except:
        # 파싱 실패 시 원본 반환
        return date_string


def create_html_email(summaries):
    """
    HTML 이메일 생성 (한국 시간 + 개선된 디자인)
    
    Args:
        summaries: 요약된 뉴스 리스트
    
    Returns:
        str: HTML 형식의 이메일 본문
    """
    # 한국 시간대 설정
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    
    # 요일 매핑
    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    weekday = weekdays[now_kst.weekday()]
    
    # 오전/오후 구분
    hour = now_kst.hour
    if hour < 12:
        period = '오전'
        display_hour = hour if hour > 0 else 12
    else:
        period = '오후'
        display_hour = hour if hour == 12 else hour - 12
    
    # 날짜 포맷: 2026년 05월 07일 목요일 오후 06:20
    formatted_date = f"{now_kst.year}년 {now_kst.month:02d}월 {now_kst.day:02d}일 {weekday} {period} {display_hour:02d}:{now_kst.minute:02d}"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
                line-height: 1.8;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7fa;
            }}
            .container {{
                background: white;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 32px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}
            .header .emoji {{
                font-size: 40px;
                margin-bottom: 10px;
            }}
            .date {{
                color: rgba(255, 255, 255, 0.9);
                font-size: 15px;
                margin-top: 12px;
                font-weight: 400;
            }}
            .content {{
                padding: 30px;
            }}
            .stats {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 20px 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
                font-size: 16px;
                font-weight: 500;
            }}
            .stats strong {{
                font-size: 24px;
                font-weight: 700;
            }}
            .news-item {{
                background: #ffffff;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 35px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .news-item:hover {{
                border-color: #667eea;
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
                transform: translateY(-2px);
            }}
            .news-header {{
                display: flex;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 15px;
            }}
            .news-number {{
                flex-shrink: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                width: 28px;
                height: 28px;
                line-height: 28px;
                text-align: center;
                border-radius: 50%;
                font-weight: 700;
                font-size: 14px;
                margin-top: 2px;
            }}
            .news-title {{
                flex: 1;
                font-size: 22px;
                font-weight: 800;
                color: #1a1a1a;
                line-height: 1.5;
                word-break: keep-all;
                margin: 0;
            }}
            .news-title a {{
                color: #1a1a1a;
                text-decoration: none;
                transition: color 0.3s ease;
            }}
            .news-title a:hover {{
                color: #667eea;
            }}
            .news-meta {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 15px 0;
                flex-wrap: wrap;
            }}
            .keyword {{
                display: inline-block;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                color: #1565c0;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 13px;
                font-weight: 600;
            }}
            .publish-date {{
                color: #868e96;
                font-size: 13px;
            }}
            .news-summary {{
                color: #495057;
                margin: 15px 0 0 0;
                line-height: 1.8;
                font-size: 15px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .footer {{
                text-align: center;
                color: #868e96;
                font-size: 13px;
                padding: 30px;
                background: #f8f9fa;
                border-top: 2px solid #e9ecef;
            }}
            .footer p {{
                margin: 8px 0;
            }}
            .footer-icon {{
                font-size: 20px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 헤더 -->
            <div class="header">
                <h1>의료 뉴스 브리핑</h1>
                <div class="date">{formatted_date}</div>
            </div>
            
            <!-- 본문 -->
    """
    
    # 뉴스 항목
    for i, news in enumerate(summaries, 1):
        # 날짜 포맷 변환
        published_date = format_korean_datetime(news.get('published_at', ''))
        
        html += f"""
                <!-- 뉴스 {i} -->
                <div class="news-item">
                    <div class="news-header">
                        <h2 class="news-title">
                            <a href="{news['url']}" target="_blank">{i}.{news['title']}</a>
                        </h2>
                    </div>
                    <div class="news-meta">
                        <span class="keyword">{news.get('keyword', 'EMR')} - </span>
                        <span class="publish-date">{published_date}</span>
                    </div>
                    <div class="news-summary">
                        {news['summary']}
                    </div>
                </div>
        """
    
    # 푸터
    html += f"""
            </div>
            
            <!-- 푸터 -->
            <div class="footer">
                <div class="footer-icon">📮</div>
                <p><strong>이 메일은 자동으로 발송되었습니다</strong></p>
                <p>EMR News Scraper v2.1</p>
                <p style="color: #adb5bd; margin-top: 15px;">
                    © {now_kst.year} EMR News Scraper. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# 수신자 리스트로 변환
def get_receiver_list():
    """환경변수에서 수신자 목록 가져오기"""
    if EMAIL_RECEIVER:
        # 쉼표로 구분하고 공백 제거
        receivers = [email.strip() for email in EMAIL_RECEIVER.split(',')]
        # 빈 문자열 제거
        receivers = [email for email in receivers if email]
        return receivers
    return []

def send_email(summaries, subject=None):
    """
    이메일로 뉴스 요약 발송 (개별 발송 - 서로 이메일 주소 안 보임)
    
    Args:
        summaries: 뉴스 요약 리스트
        subject: 이메일 제목 (None이면 자동 생성)
    
    Returns:
        dict: 발송 결과
    """
    try:
        # 수신자 목록 가져오기
        receiver_list = get_receiver_list()
        
        if not receiver_list:
            logger.error("❌ 수신자 이메일이 설정되지 않았습니다.")
            return {
                'success': False,
                'message': 'EMAIL_RECEIVER 환경변수를 확인하세요'
            }
        
        # 제목 생성
        if subject is None:
            subject = f"🏥 EMR 뉴스 브리핑 ({datetime.now().strftime('%Y-%m-%d')})"
        
        # HTML 본문 (한 번만 생성)
        html_body = create_html_email(summaries)
        
        # 각 수신자에게 개별 발송
        success_count = 0
        failed_list = []
        
        logger.info(f"📧 이메일 개별 발송 시작: {len(receiver_list)}명")
        
        # SMTP 연결을 한 번만 하고 여러 번 전송 (최적화)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            
            for receiver in receiver_list:
                try:
                    # 각 수신자별로 새로운 메시지 생성
                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = subject
                    msg['From'] = EMAIL_SENDER
                    msg['To'] = receiver  # 한 명씩만!
                    
                    html_part = MIMEText(html_body, 'html', 'utf-8')
                    msg.attach(html_part)
                    
                    server.send_message(msg)
                    logger.info(f"   ✅ {receiver} 발송 완료")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"   ❌ {receiver} 발송 실패: {str(e)}")
                    failed_list.append(receiver)
        
        # 결과 정리
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
            logger.error(f"❌ 모든 이메일 발송 실패")
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