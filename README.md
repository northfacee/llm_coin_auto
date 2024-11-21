# Crypto Trading Analysis Bot

실시간으로 암호화폐 시장을 분석하고 투자 결정을 제안하는 AI 기반 트레이딩 봇입니다.
빗썸버전입니다.

## 기능

- 비트코인뿐만 아닌 다양한 코인 매매가능
- LangSmith를 통한 분석 결과 모니터링
- Langgraph를 통한 에이전트 (뉴스 에이전트, 가격 에이전트)
- Streamlit을 통해 매매 모니터링 가능

## 매매과정

1. 크게 뉴스 에이전트, 가격 에이전트, 최종 에이전트 3개가 있습니다.
2. 뉴스 에이전트의 경우 네이버 api를 사용하여 search_keywords에 키워드들을 넣으면 그 키워드의 뉴스들을 수집하여 이 정보를 토대로 결정합니다.
3. 가격 에이전트의 경우 빗썸 api를 사용하여 1m, 3m, 5m, 10m, 15m, 30m의 데이터들을 다양한 지표로 변환하여 이 정보를 토대로 결정합니다.
4. 최종 에이전트는 뉴스, 가격 에이전트의 정보를 종합적으로 판단하여 결정합니다.
5. 매매내역은 Streamlit을 통해 모니터링 가능합니다.

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/northfacee/llm_coin_auto.git
cd test
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
`.env` 파일을 생성하고 다음 값들을 설정하세요:
```
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key

NAVER_CLIENT_ID=your_naver_api_key
NAVER_CLIENT_SECRET=your_naver_api_secret_key

BITHUMB_API_KEY=your_bithumb_api_key
BITHUMB_API_SECRET=your_bithumb_api_secret_key

BITHUMB_API_KEY_V2=your_bithumb_v2_api_key
BITHUMB_API_SECRET_V2=your_bithumb_v2_api_secret_key

INVESTMENT= your_initial_money #예시 1000000
COIN=your_coin #예시 BTC, DOGE
```

## 실행 방법

```bash
python decision.py
streamlit run app.py
```

## 주의사항

- 이 봇은 교육 및 연구 목적으로 제작되었습니다.
- 실제 트레이딩에 사용할 경우 발생하는 수익,손실에 대해 절대 절대 책임지지 않습니다.
- 계속 실행하기 위해서는 클라우드 서비스를 사용하는게 좋습니다.

## 추후 추가 사항

- 현재는 비트코인만 매매하지만, 상위 거래량으로 순위를 매겨 다양한 코인 자동매매 가능성
- 나스닥, 공포지수 등도 포함하여 분석
- 업비트나 바이낸스 등 추가해볼 예정임

## 기타

- 기술적 지표의 자세한 값은 Langsmith로 확인하거나 price_collector.py를 실행해서 확인 할 수 있습니다.