"""
OpenAI API를 사용한 뉴스 요약 모듈
"""
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 모델 설정
MODEL = "gpt-4o-mini"  # 저렴하고 빠른 모델


def summarize_article(article: Dict, max_length: int = 150) -> Dict:
    """
    단일 기사 요약
    
    Args:
        article: {
            'title': str,
            'content': str,  # description
            'url': str,
            'keyword': str
        }
        max_length: 요약 최대 길이 (기본 150자)
    
    Returns:
        {
            'title': str,
            'summary': str,
            'url': str,
            'keyword': str,
            'tokens_used': int
        }
    """
    
    try:
        prompt = f"""
다음 뉴스 기사를 {max_length}자 이내로 요약해주세요.
핵심 내용만 간결하게 정리하되, 의료/헬스케어 관점에서 중요한 정보를 포함하세요.

제목: {article['title']}
내용: {article['content']}

요약:
"""
        
        logger.info(f"📝 요약 중: {article['title'][:30]}...")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 의료/헬스케어 분야 전문 기자입니다. 기사를 간결하고 정확하게 요약합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # 일관성 있는 요약
            max_tokens=200
        )
        
        summary = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        logger.info(f"✅ 요약 완료: {len(summary)}자 (토큰: {tokens_used})")
        
        return {
            'title': article['title'],
            'summary': summary,
            'url': article['url'],
            'keyword': article.get('keyword', ''),
            'published_at': article.get('published_at', ''),
            'tokens_used': tokens_used
        }
        
    except Exception as e:
        logger.error(f"❌ 요약 실패: {e}")
        return {
            'title': article['title'],
            'summary': article['content'][:150] + "...",  # 실패 시 원문 일부
            'url': article['url'],
            'keyword': article.get('keyword', ''),
            'published_at': article.get('published_at', ''),
            'tokens_used': 0,
            'error': str(e)
        }


def summarize_batch(articles: List[Dict], delay: float = 1.0) -> List[Dict]:
    """
    여러 기사 일괄 요약
    
    Args:
        articles: 기사 리스트
        delay: API 호출 간 대기 시간 (초)
    
    Returns:
        요약 결과 리스트
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🤖 {len(articles)}개 기사 요약 시작")
    logger.info(f"{'='*60}\n")
    
    summaries = []
    total_tokens = 0
    
    for i, article in enumerate(articles, 1):
        logger.info(f"진행: {i}/{len(articles)}")
        
        summary = summarize_article(article)
        summaries.append(summary)
        
        total_tokens += summary.get('tokens_used', 0)
        
        # API 호출 제한 방지
        if i < len(articles):
            time.sleep(delay)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 요약 완료!")
    logger.info(f"총 토큰: {total_tokens:,}")
    logger.info(f"예상 비용: ${total_tokens * 0.00015:.4f} USD")
    logger.info(f"{'='*60}\n")
    
    return summaries


def test_summarizer():
    """
    테스트 함수
    """
    test_article = {
        'title': 'AI 기반 EMR 시스템 도입 확대',
        'content': '국내 주요 병원들이 인공지능 기반 전자의무기록(EMR) 시스템을 도입하고 있다. 서울대병원과 삼성서울병원은 AI를 활용해 의료진의 기록 부담을 줄이고 진료 효율성을 높이는 데 성공했다.',
        'url': 'https://example.com/news1',
        'keyword': 'EMR'
    }
    
    print("\n" + "="*60)
    print("🧪 Summarizer 테스트")
    print("="*60)
    
    result = summarize_article(test_article)
    
    print(f"\n✅ 결과:")
    print(f"제목: {result['title']}")
    print(f"요약: {result['summary']}")
    print(f"토큰: {result['tokens_used']}")
    print(f"비용: ${result['tokens_used'] * 0.00015:.4f} USD")
    print("\n" + "="*60)


# 테스트 실행
if __name__ == "__main__":
    test_summarizer()