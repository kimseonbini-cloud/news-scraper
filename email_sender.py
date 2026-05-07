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


def create_html_email(summaries: list) -> str:
    """
    HTML 형식의 이메일 본문 생성
    
    Args:
        summaries: [{title, summary, url, keyword}, ...]
    
    Returns:
        HTML 문자열
    """
    
    # 헤더
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .date {{
                color: #eee;
                font-size: 14px;
                margin-top: 10px;
            }}
            .news-item {{
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .news-title {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .news-summary {{
                color: #555;
                margin-bottom: 15px;
                line-height: 1.8;
            }}
            .news-meta {{
                font-size: 12px;
                color: #888;
                margin-bottom: 10px;
            }}
            .keyword {{
                display: inline-block;
                background: #e3f2fd;
                color: #1976d2;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-right: 5px;
            }}
            .link {{
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }}
            .link:hover {{
                text-decoration: underline;
            }}
            .footer {{
                text-align: center;
                color: #888;
                font-size: 12px;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
            .stats {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 EMR 뉴스 브리핑</h1>
            <div class="date">{datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</div>
        </div>
        
        <div class="stats">
            <strong>📊 오늘의 요약</strong><br>
            총 <strong>{len(summaries)}개</strong>의 주요 뉴스를 수집했습니다.
        </div>
    """
    
    # 뉴스 항목
    for i, news in enumerate(summaries, 1):
        html += f"""
        <div class="news-item">
            <div class="news-title">
                {i}. {news['title']}
            </div>
            <div class="news-meta">
                <span class="keyword">{news.get('keyword', 'EMR')}</span>
                {news.get('published_at', '')}
            </div>
            <div class="news-summary">
                {news['summary']}
            </div>
            <a href="{news['url']}" class="link" target="_blank">
                🔗 원문 보기
            </a>
        </div>
        """
    
    # 푸터
    html += """
        <div class="footer">
            <p>📮 이 메일은 자동으로 발송되었습니다.</p>
            <p>EMR News Scraper v1.0</p>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(summaries: list, subject: str = None) -> dict:
    """
    이메일 발송
    
    Args:
        summaries: 요약 리스트
        subject: 이메일 제목 (기본값: 자동 생성)
    
    Returns:
        {
            'success': bool,
            'message': str
        }
    """
    
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.error("❌ 이메일 설정이 없습니다!")
        return {
            'success': False,
            'message': '.env에서 EMAIL_SENDER, EMAIL_PASSWORD 설정 필요'
        }
    
    if not EMAIL_RECEIVER:
        logger.warning("⚠️ 수신자 이메일이 없습니다. 발신자에게 전송합니다.")
        receiver = EMAIL_SENDER
    else:
        receiver = EMAIL_RECEIVER
    
    try:
        # 제목 생성
        if subject is None:
            subject = f"🏥 EMR 뉴스 브리핑 ({datetime.now().strftime('%Y-%m-%d')})"
        
        # 이메일 구성
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = receiver
        
        # HTML 본문
        html_body = create_html_email(summaries)
        html_part = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(html_part)
        
        # SMTP 연결 및 전송
        logger.info(f"📧 이메일 발송 중: {receiver}")
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # TLS 보안 연결
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✅ 이메일 발송 완료!")
        
        return {
            'success': True,
            'message': f'{receiver}로 전송 완료'
        }
        
    except smtplib.SMTPAuthenticationError:
        logger.error("❌ 로그인 실패! 이메일/비밀번호를 확인하세요.")
        return {
            'success': False,
            'message': 'SMTP 인증 실패 (앱 비밀번호 확인 필요)'
        }
        
    except Exception as e:
        logger.error(f"❌ 이메일 발송 실패: {e}")
        return {
            'success': False,
            'message': str(e)
        }


def send_test_email():
    """
    테스트 이메일 발송
    """
    test_summaries = [
        {
            'title': '테스트: AI 기반 EMR 시스템 도입',
            'summary': '국내 주요 병원들이 인공지능 기반 전자의무기록 시스템을 도입하고 있습니다. 이를 통해 의료 서비스의 질이 향상될 것으로 기대됩니다.',
            'url': 'https://example.com/news1',
            'keyword': 'EMR',
            'published_at': datetime.now().strftime('%Y-%m-%d')
        },
        {
            'title': '테스트: 디지털헬스케어 시장 급성장',
            'summary': '디지털헬스케어 시장이 연평균 20% 이상 성장하고 있습니다. 특히 원격의료와 AI 진단 분야가 주목받고 있습니다.',
            'url': 'https://example.com/news2',
            'keyword': '디지털헬스케어',
            'published_at': datetime.now().strftime('%Y-%m-%d')
        }
    ]
    
    return send_email(test_summaries, subject="🧪 테스트: EMR 뉴스 브리핑")


# ====================================
# 테스트 코드
# ====================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📧 이메일 발송 테스트")
    print("="*60)
    
    result = send_test_email()
    
    if result['success']:
        print(f"\n✅ 성공: {result['message']}")
        print("\n📮 이메일을 확인해보세요!")
    else:
        print(f"\n❌ 실패: {result['message']}")
    
    print("\n" + "="*60)