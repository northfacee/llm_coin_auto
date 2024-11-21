from typing import TypedDict, Sequence
from datetime import datetime, timedelta
import json
import os
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from langchain_core.tools import tool
from langsmith import Client
import langsmith
from dotenv import load_dotenv
from database_manager import DatabaseManager
from price_collector import BithumbTrader
from news_collector import NaverNewsCollector
from trading import BithumbTradeExecutor
import time
import traceback  # traceback 모듈 추가
from pprint import pprint

# 환경 변수 및 LangSmith 설정
load_dotenv()
try:
    INVESTMENT = float(os.getenv('INVESTMENT'))
except TypeError:
    raise ValueError("INVESTMENT 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

SYMBOL = os.getenv('COIN')
if not SYMBOL:
    raise ValueError("COIN 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "bitcoin_agent"


# 상태 타입 정의
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next_step: str
    results: dict
    market_data: dict
    symbol: str

# 글로벌 인스턴스 초기화
db_manager = DatabaseManager()
trader = BithumbTrader()
trade_executor = BithumbTradeExecutor()
langsmith_client = Client()

def collect_latest_news():
    """최신 뉴스를 수집하고 저장"""
    try:
        local_news_collector = NaverNewsCollector()
        total_saved = 0
        
        for keyword in local_news_collector.search_keywords:
            news_response = local_news_collector.collect_news(keyword, display=10)
            if news_response:
                news_items = local_news_collector.process_news_data(news_response['items'])
                if local_news_collector.save_news(news_items):
                    total_saved += len(news_items)
        
        print(f"총 {total_saved}개의 새로운 뉴스 기사가 저장되었습니다.")
        
    except Exception as e:
        print(f"뉴스 수집 중 오류 발생: {e}")

def get_market_data_once(state: AgentState) -> dict:
    """시장 데이터를 안전하게 수집"""
    try:
        # 이미 market_data가 있다면 재사용
        if state.get('market_data') and isinstance(state['market_data'], dict):
            return state['market_data']
            
        if not trader.analyzer.data_queue.empty():
            market_data = trader.analyzer.data_queue.get()
        else:
            market_data = trader.collect_market_data()
        
        if not isinstance(market_data, dict):
            print("Warning: market_data is not a dictionary")
            market_data = {'error': 'Invalid data format'}
            
        if 'current_price' in market_data:
            if not isinstance(market_data['current_price'], dict):
                current_price_value = market_data['current_price']
                market_data['current_price'] = {
                    'closing_price': float(current_price_value),
                    'opening_price': float(current_price_value),
                    'max_price': float(current_price_value),
                    'min_price': float(current_price_value)
                }
        
        state['market_data'] = market_data
        return market_data
        
    except Exception as e:
        print(f"Error in get_market_data_once: {str(e)}")
        return {
            'timestamp': datetime.now().isoformat(),
            'market': f'{SYMBOL}_KRW',
            'current_price': {'closing_price': 0}
        }

@tool
def get_recent_news(dummy: str = "") -> str:
    """최근 24시간 동안의 암호화폐 관련 뉴스를 가져옵니다."""
    with langsmith.trace(
        name="get_recent_news",
        project_name="bitcoin_agent",
        tags=["news", "data-collection"]
    ):
        # 최신 뉴스 수집을 먼저 실행
        collect_latest_news()
        
        news_df = db_manager.get_recent_news_limit()
        #print(news_df)
        if news_df.empty:
            return "최근 뉴스가 없습니다."
        
        news_list = []
        for _, row in news_df.iterrows():
            news_list.append({
                'title': row['title'],
                'description': row['description'],
                'pub_date': row['pub_date'].isoformat() if hasattr(row['pub_date'], 'isoformat') else str(row['pub_date'])
            })
        return json.dumps(news_list, ensure_ascii=False)

def news_analysis_agent(state: AgentState) -> AgentState:
    """뉴스만을 분석하여 투자 제안을 하는 에이전트"""
    with langsmith.trace(
        name="news_analysis_agent",
        project_name="bitcoin_agent",
        tags=["news", "analysis"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        news_data = get_recent_news.invoke("")
        
        prompt = f"""당신은 암호화폐 시장의 뉴스 분석 전문가입니다.
                    단기 가격 변동에 즉각적 영향을 미칠 수 있는 뉴스에 집중하여 분석해주세요.

                    분석 우선순위:
                    1. 규제/정책 관련 뉴스 (영향력 가중치: 35%)
                    2. 기관 투자자 동향 (영향력 가중치: 25%)
                    3. 기술적 발전/업데이트 소식 (영향력 가중치: 20%)
                    4. 시장 심리 관련 뉴스 (영향력 가중치: 20%)

                    뉴스 데이터:
                    {news_data}
                    
                    다음 형식으로 상세한 분석 결과를 제공해주세요:
                    1. 투자결정: (매수/매도/관망)
                    2. 투자비중: (0-100%)
                    3. 결정 이유:
                    - 주요 영향 뉴스 (영향력 순위별)
                    - 예상되는 시장 반응
                    - 영향 지속 기간 예측
                    4. 리스크 요소:
                    - 단기 리스크
                    - 반대 시나리오 가능성
                    5. 종합 뉴스 영향도: (매우 부정적/-2 ~ 매우 긍정적/+2)
                    """
        
        response = llm.invoke(prompt)
        timestamp = datetime.now()
        
        if 'results' not in state:
            state['results'] = {}
        
        state['results']['news_analysis'] = {
            'analysis': response.content,
            'timestamp': timestamp.isoformat()
        }
        
        db_manager.save_news_analysis(
            timestamp=timestamp,
            analysis_text=response.content
        )
        
        return state

def price_analysis_agent(state: AgentState) -> AgentState:
    """기술적 지표와 시장 데이터를 분석하여 투자 제안을 하는 에이전트"""
    with langsmith.trace(
        name="price_analysis_agent",
        project_name="bitcoin_agent",
        tags=["price", "analysis"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        market_data = get_market_data_once(state)
        
        def safe_get_nested(data, *keys, default='N/A'):
            try:
                result = data
                for key in keys:
                    if isinstance(result, dict):
                        result = result.get(key, default)
                    elif isinstance(result, (list, tuple)) and isinstance(key, int):
                        result = result[key] if 0 <= key < len(result) else default
                    else:
                        return default
                return result if result is not None else default
            except Exception as e:
                print(f"Error accessing {keys}: {str(e)}")
                return default

        def format_number(value, default='N/A'):
            if value == 'N/A':
                return value
            try:
                if isinstance(value, (int, float)):
                    return f"{value:,.2f}"
                return str(default)
            except:
                return str(default)
        
        try:
            # 모든 시간대의 분석 데이터 가져오기
            time_periods = ['1m', '3m', '5m', '10m', '15m', '30m']
            analysis_data = {}
            
            for period in time_periods:
                analysis_data[period] = safe_get_nested(market_data, 'analysis', period, default={})
                
            current_price = safe_get_nested(market_data, 'current_price', 'closing_price', default=0)
            
            # 각 시간대별 분석 데이터 구성
            analysis_by_period = {}
            for period in time_periods:
                period_data = analysis_data[period]
                
                # 이동평균선 데이터 처리
                moving_averages = period_data.get('moving_averages', {})
                
                analysis_by_period[period] = {
                    'MA': {
                        'MA5': format_number(safe_get_nested(moving_averages, 5)),
                        'MA10': format_number(safe_get_nested(moving_averages, 10)),
                        'MA20': format_number(safe_get_nested(moving_averages, 20)),
                        'MA50': format_number(safe_get_nested(moving_averages, 50)),
                        'MA200': format_number(safe_get_nested(moving_averages, 200))
                    },
                    'RSI': format_number(safe_get_nested(period_data, 'rsi')),
                    'Stochastic': {
                        'K': format_number(safe_get_nested(period_data, 'stochastic', 0)),
                        'D': format_number(safe_get_nested(period_data, 'stochastic', 1))
                    },
                    'BB': {
                        'Upper': format_number(safe_get_nested(period_data, 'bollinger_bands', 0)),
                        'Middle': format_number(safe_get_nested(period_data, 'bollinger_bands', 1)),
                        'Lower': format_number(safe_get_nested(period_data, 'bollinger_bands', 2))
                    },
                    'EMA': {
                        'EMA12': format_number(safe_get_nested(period_data, 'ema', '12')),
                        'EMA26': format_number(safe_get_nested(period_data, 'ema', '26'))
                    },
                    'DMI': {
                        '+DI': format_number(safe_get_nested(period_data, 'dmi', 0)),
                        '-DI': format_number(safe_get_nested(period_data, 'dmi', 1)),
                        'ADX': format_number(safe_get_nested(period_data, 'dmi', 2))
                    },
                    'ATR': format_number(safe_get_nested(period_data, 'atr')),
                    'OBV': format_number(safe_get_nested(period_data, 'obv')),
                    'VWAP': format_number(safe_get_nested(period_data, 'vwap')),
                    'MFI': format_number(safe_get_nested(period_data, 'mfi')),
                    'Williams_R': format_number(safe_get_nested(period_data, 'williams_r')),
                    'CCI': format_number(safe_get_nested(period_data, 'cci')),
                    'Change_Rate': format_number(safe_get_nested(period_data, 'change_rate'))
                }
            
            # 프롬프트 구성
            prompt = f"""당신은 암호화폐 기술적 분석 전문가입니다. 
            다음 데이터를 기반으로 투자 결정을 내려주세요.
            
            현재가: {format_number(current_price)}원
            
            === 각 시간대별 변동률 ===
            1분: {analysis_by_period['1m']['Change_Rate']}%
            3분: {analysis_by_period['3m']['Change_Rate']}%
            5분: {analysis_by_period['5m']['Change_Rate']}%
            10분: {analysis_by_period['10m']['Change_Rate']}%
            15분: {analysis_by_period['15m']['Change_Rate']}%
            30분: {analysis_by_period['30m']['Change_Rate']}%
            """
            
            # 각 시간대별 데이터를 프롬프트에 추가
            for period in time_periods:

                period_name = {
                    '1m': '1분',
                    '3m': '3분',
                    '5m': '5분',
                    '10m': '10분',
                    '15m': '15분',
                    '30m': '30분',
                }[period]
                
                data = analysis_by_period[period]
                prompt += f"""
                === {period_name} 기준 지표 ===
                - RSI(14): {data['RSI']}
                - Stochastic K/D: {data['Stochastic']['K']}/{data['Stochastic']['D']}
                - 볼린저 밴드: 
                  상단: {data['BB']['Upper']}
                  중간: {data['BB']['Middle']}
                  하단: {data['BB']['Lower']}
                - 이동평균선:
                  MA5: {data['MA']['MA5']}
                  MA10: {data['MA']['MA10']}
                  MA20: {data['MA']['MA20']}
                  MA50: {data['MA']['MA50']}
                - EMA:
                EMA12: {data['EMA']['EMA12']}
                EMA26: {data['EMA']['EMA26']}
                - ATR: {data['ATR']}
                - VWAP: {data['VWAP']}
                - MFI: {data['MFI']}
                - Williams %R: {data['Williams_R']}
                - CCI: {data['CCI']}
                - 변동률: {data['Change_Rate']}%
                """
            
            prompt += f"""
            [시장 분석 기준]
            1. 추세 강도 판단
               - 이동평균선 배열
               - 추세선 저항/지지
               - 추세 모멘텀 지속성
            
            2. 단기 반전 신호
               - RSI, Stochastic 다이버전스
               - 볼린저 밴드 침투
               - 거래량 급증 구간
            
            3. 매매 시점 정밀도
               - 15분봉 이하 단기 차트 우선
               - 복합 지표 확증 신호
               - 거래량 프로필 기반 가격대 분석
            
            4. 리스크 관리
               - 변동성 기반 손절가
               - 예상 손익비
               - 포지션 진입 타이밍
            
            다음 형식으로 분석 결과를 제공해주세요:
            1. 투자 판단
               - 투자결정: (매수/매도/관망)
               - 투자비중: (0-100%)
               - 목표가/손절가
               - 예상 진입 유지 시간
            
            2. 주요 시그널 분석
               - 단기(1-5분): 주요 전환점/강도
               - 중기(10-30분): 추세 방향/강도
               - 거래량 특이점
               - 주목할 기술적 패턴
            
            3. 리스크 평가
               - 반전 가능성
               - 변동성 수준
               - 거래량 리스크
               - 추천 레버리지 배수
            
            4. 종합 점수: (-5 ~ +5)
               음수: 하락 가능성
               양수: 상승 가능성
               절대값: 신뢰도
               """
            
            response = llm.invoke(prompt)
            timestamp = datetime.now()
            
            state['results']['price_analysis'] = {
                'analysis': response.content,
                'timestamp': timestamp.isoformat()
            }
            
            db_manager.save_price_analysis(
                timestamp=timestamp,
                current_price=current_price,
                analysis_text=response.content
            )
            
        except Exception as e:
            print(f"Error in price analysis: {str(e)}")
            state['results']['price_analysis'] = {
                'analysis': "기술적 분석 중 오류가 발생했습니다. 다음 분석을 기다려주세요.",
                'timestamp': datetime.now().isoformat()
            }
        
        return state
    
def final_decision_agent(state: AgentState) -> AgentState:
    with langsmith.trace(
        name="final_decision_agent",
        project_name="bitcoin_agent",
        tags=["final", "decision"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        market_data = get_market_data_once(state)
        current_price = 0
        
        try:
            if isinstance(market_data.get('current_price'), (int, float)):
                current_price = market_data['current_price']
            elif isinstance(market_data.get('current_price'), dict):
                current_price = market_data['current_price'].get('closing_price', 0)
            else:
                current_price = 0
                print("Warning: Unable to determine current price format")
        except Exception as e:
            print(f"Error extracting current price: {e}")

        try:
            position = trade_executor.get_current_position()
            avg_price = position.get('avg_price', 0)
            quantity = position.get('total_quantity', 0)
            investment = position.get('total_investment', 0)
            
            profit_rate = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            position_text = f"""현재 보유 포지션:
            - 평균 매수가: {avg_price:,.0f}원
            - 현재 수익률: {profit_rate:.2f}%
            - 총 투자금액: {investment:,.0f}원
            - 보유 수량: {quantity:.8f} {SYMBOL}"""

        except Exception as e:
            print(f"포지션 정보 조회 실패: {e}")
            position_text = "현재 보유 중인 포지션이 없습니다."

        try:
            position = trade_executor.get_current_position()
            avg_price = position.get('avg_price', 0)
            quantity = position.get('total_quantity', 0)
            investment = position.get('total_investment', 0)
            
            profit_rate = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            # 수익률이 +1% 초과 또는 -1% 초과인 경우 자동 매도 결정
            if abs(profit_rate) > 1 and quantity > 0:
                print(f"수익률 {profit_rate:.2f}%로 임계값(±1%) 초과. 자동 매도 실행.")
                auto_decision = {
                    "decision": "SELL",
                    "percentage": 100,  # 전량 매도
                    "price_score": 0,
                    "news_score": 0,
                    "analysis": {
                        "market_trend": "NEUTRAL",
                        "market_status": f"수익률 {profit_rate:.2f}% 도달로 인한 자동 매도",
                        "risk_level": "LOW"
                    },
                    "signals": {
                        "technical": "NEUTRAL",
                        "news": "NEUTRAL",
                        "trend": "SIDEWAYS"
                    },
                    "reason": f"수익률 {profit_rate:.2f}%가 임계값(±1%)을 초과하여 자동 매도 실행"
                }
                
                state['results']['final_decision'] = {
                    'decision': auto_decision,
                    'timestamp': datetime.now().isoformat()
                }

                                # JSON으로 저장할 때 한글 인코딩 처리
                db_manager.save_final_decision(
                    timestamp=datetime.now(),
                    current_price=current_price,
                    analysis_text=json.dumps(auto_decision, ensure_ascii=False)
                )
                
                return state
            
            position_text = f"""현재 보유 포지션:
            - 평균 매수가: {avg_price:,.0f}원
            - 현재 수익률: {profit_rate:.2f}%
            - 총 투자금액: {investment:,.0f}원
            - 보유 수량: {quantity:.8f} {SYMBOL}"""

        except Exception as e:
            print(f"포지션 정보 조회 실패: {e}")
            position_text = "현재 보유 중인 포지션이 없습니다."

        prompt = f"""
        당신은 암호화폐 투자 전문가이자 리스크 관리자입니다.
        스캘핑과 데이트레이딩 관점에서 단기 수익 기회를 포착하는데 집중합니다.
        아래 시장분석에서 뉴스분석과 가격분석을 종합해서 결정해주세요.

        [중요 제한사항]
        - 투자 비중은 반드시 0-30% 범위 내에서만 결정해야 합니다
        - 30%를 초과하는 투자 비중은 절대 제안하지 마세요
        - HOLD 결정시 반드시 투자 비중은 0%여야 합니다

        주요 투자 원칙:
        - 수익률이 +1% 초과 또는 -1% 초과 시 즉시 전량 매도
        - 투자 비중은 0-30% 범위로 엄격히 제한 (절대 초과 불가)
        - 투자 결정 가중치: 가격 분석 85%, 뉴스 분석 15%

        시장 현황:
        현재가: {current_price:,.0f}원

        포지션 분석:
        {position_text}

        시장 분석:
        [뉴스 분석]
        {state['results']['news_analysis']['analysis']}

        [기술적 분석]
        {state['results']['price_analysis']['analysis']}


        아래의 정확한 JSON 형식으로만 결과를 내주세요. 다른 텍스트를 추가하지 마세요:

        {{
            "decision": "BUY/SELL/HOLD 중 하나만 선택",
            "percentage": "반드시 0-30 사이의 정수만 가능 (30 초과 절대 불가)",
            
            "analysis": {{
                "market_trend": "강세/약세/중립 중 하나만 선택",
                "market_status": "현재 시장 상황 간단 요약(50자 이내)",
                "risk_level": "상/중/하 중 하나만 선택"
            }},
            "signals": {{
                "technical": "강력매수/매수/중립/매도/강력매도 중 하나만 선택",
                "news": "긍정/부정/중립 중 하나만 선택",
                "trend": "상승/하락/횡보 중 하나만 선택"
            }},
            "reason": "투자 결정의 주된 이유(100자 이내)"
        }}

        [필수 규칙 - 위반 시 결과 무효]
        1. decision은 반드시 "BUY", "SELL", "HOLD" 중 하나여야 합니다 (대문자만).
        2. percentage는 반드시 0에서 30 사이의 정수여야 합니다 (30 초과 절대 불가).
        3. HOLD 결정 시 percentage는 반드시 0이어야 합니다.
        4. BUY 또는 SELL 결정 시 percentage는 1-30 사이의 정수여야 합니다.
        5. 모든 문자열은 정확히 제시된 선택지 중에서만 선택해야 합니다.
        6. 모든 점수는 정수값이어야 합니다.
        7. 모든 문자열은 큰따옴표로 감싸야 합니다.
        8. JSON 형식을 정확히 지켜야 합니다.
        9. 추가 설명이나 텍스트를 포함하지 마세요.
        """

        response = llm.invoke(prompt)
        timestamp = datetime.now()
        
        try:
            # 응답에서 JSON 부분만 추출 (추가 텍스트가 있을 경우를 대비)
            response_text = response.content.strip()
            if response_text.startswith('{') and response_text.endswith('}'):
                decision_json = json.loads(response_text)
            else:
                # JSON이 아닌 텍스트가 포함된 경우, JSON 부분만 추출 시도
                import re
                json_match = re.search(r'\{[^{]*\}', response_text)
                if json_match:
                    decision_json = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError("No valid JSON found", response_text, 0)
            
            state['results']['final_decision'] = {
                'decision': decision_json,
                'timestamp': timestamp.isoformat()
            }

                        # JSON으로 저장할 때 한글 인코딩 처리
            db_manager.save_final_decision(
                timestamp=datetime.now(),
                current_price=current_price,
                analysis_text=json.dumps(decision_json, ensure_ascii=False, indent=2)
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            default_decision = {
                'decision': 'HOLD',
                'percentage': 0,
                'price_score': 0,
                'news_score': 0,
                'analysis': {
                    'market_trend': 'NEUTRAL',
                    'market_status': '분석 실패',
                    'risk_level': 'HIGH'
                },
                'signals': {
                    'technical': 'NEUTRAL',
                    'news': 'NEUTRAL',
                    'trend': 'SIDEWAYS'
                },
                'reason': 'JSON 파싱 오류로 인한 자동 HOLD 결정'
            }
            
            state['results']['final_decision'] = {
                'decision': default_decision,
                'timestamp': datetime.now().isoformat()
            }
            
            db_manager.save_final_decision(
                timestamp=datetime.now(),
                current_price=current_price,
                analysis_text=json.dumps(default_decision, ensure_ascii=False, indent=2)
            )
        
        return state
    
def execute_trading_decision(state: AgentState) -> None:
    """거래 결정을 실행하는 함수"""
    try:
        if 'final_decision' not in state['results']:
            print("최종 결정이 없어 거래를 실행할 수 없습니다.")
            return

        timestamp = datetime.now()

        # 오래된 미체결 주문 취소 처리
        trade_executor.check_and_cancel_old_orders()
        
        try:
            final_decision = state['results']['final_decision']
            current_price = float(state['market_data']['current_price']['closing_price'])
            
            # 거래 실행
            result = trade_executor.execute_trade(
                decision=final_decision,
                max_investment=INVESTMENT,
                current_price=current_price
            )
            
            # 거래 결과 출력
            print("\n=== 거래 실행 결과 ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # HOLD 타입일 경우 별도 처리
            if result.get('type') == 'HOLD':
                db_manager.save_trade_execution(
                    timestamp=timestamp,
                    trade_type='HOLD',
                    quantity=0,
                    price=current_price,  # 현재가 사용
                    total_amount=0,
                    order_id=f"HOLD_{timestamp.strftime('%Y%m%d%H%M%S')}"
                )
            # 실제 거래 케이스 처리
            elif result['status'] == 'SUCCESS':
                db_manager.save_trade_execution(
                    timestamp=timestamp,
                    trade_type=result['type'],
                    quantity=result.get('quantity', 0),
                    price=result.get('price', current_price),
                    total_amount=result.get('total_amount', 0),
                    order_id=result.get('order_id', f"{result['type']}_{timestamp.strftime('%Y%m%d%H%M%S')}")
                )
            # 에러 케이스 처리
            else:
                db_manager.save_trade_execution(
                    timestamp=timestamp,
                    trade_type=result.get('type', 'ERROR'),
                    quantity=0,
                    price=current_price,  # 현재가 사용
                    total_amount=0,
                    order_id=f"ERROR_{timestamp.strftime('%Y%m%d%H%M%S')}"
                )
                
        except Exception as e:
            print(f"거래 실행 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            
            db_manager.save_trade_execution(
                timestamp=timestamp,
                trade_type='ERROR',
                quantity=0,
                price=current_price,  # 현재가 사용
                total_amount=0,
                order_id=f"ERROR_{timestamp.strftime('%Y%m%d%H%M%S')}"
            )
            
    except Exception as e:
        print(f"거래 실행 함수 전체 오류: {e}")
        print("오류 세부 정보:", traceback.format_exc())


def create_trading_workflow() -> Graph:
    """트레이딩 워크플로우 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("news_analysis", news_analysis_agent)
    workflow.add_node("price_analysis", price_analysis_agent)
    workflow.add_node("final_decision", final_decision_agent)
    
    # 엣지 추가 (실행 순서 정의)
    workflow.add_edge("news_analysis", "price_analysis")
    workflow.add_edge("price_analysis", "final_decision")
    
    # 시작점과 종료점 설정
    workflow.set_entry_point("news_analysis")
    workflow.set_finish_point("final_decision")
    
    return workflow.compile()

def run_trading_analysis():
    """트레이딩 분석 실행"""
    try:
        print("트레이딩 분석 시작 (LangSmith 모니터링 활성화)")
        app = create_trading_workflow()
        config = {
            "messages": [], 
            "next_step": "news_analysis",
            "results": {},
            "market_data": {}
        }
        
        with langsmith.trace(
            name="complete_trading_analysis",
            project_name="bitcoin_agent",
            tags=["workflow", "complete"]
        ) as tracer:
            result = app.invoke(config)
            
            if 'results' in result:
                print("\n=== 분석 결과 ===")
                if 'news_analysis' in result['results']:
                    print("\n[뉴스 분석]")
                    print(result['results']['news_analysis']['analysis'])
                
                if 'price_analysis' in result['results']:
                    print("\n[가격 분석]")
                    print(result['results']['price_analysis']['analysis'])
                
                if 'final_decision' in result['results']:
                    print("\n[최종 결정]")
                    pprint(result['results']['final_decision']['decision'])
                    
                    # 거래 실행 추가
                    execute_trading_decision(result)
            else:
                print("분석 결과가 없습니다.")
        
    except Exception as e:
        print(f"분석 실행 중 오류 발생: {e}")
        raise e

def run_continuous_analysis():
    """30분마다 트레이딩 분석을 실행하는 연속 실행 함수"""
    WAIT_MINUTES = 1
    WAIT_SECONDS = WAIT_MINUTES * 60  # 30분을 초로 변환
    
    print("연속 트레이딩 분석 시작...")
    print(f"실행 간격: {WAIT_MINUTES}분")
    print("Ctrl+C를 눌러서 프로그램을 종료할 수 있습니다.")
    
    while True:
        try:
            # 현재 시간 출력
            current_time = datetime.now()
            print(f"\n{'='*50}")
            print(f"새로운 분석 시작 시간: {current_time}")
            print(f"{'='*50}\n")
            
            # 트레이딩 분석 실행
            run_trading_analysis()
            
            # 다음 실행까지 대기
            next_run_time = current_time + timedelta(minutes=WAIT_MINUTES)
            print(f"\n다음 분석 예정 시간: {next_run_time}")
            print(f"다음 분석까지 {WAIT_MINUTES}분 대기 중...")
            
            # 지정된 시간만큼 대기
            time.sleep(WAIT_SECONDS)
            
        except KeyboardInterrupt:
            print("\n프로그램이 사용자에 의해 종료되었습니다.")
            break
        except Exception as e:
            print(f"예기치 않은 오류 발생: {e}")
            print(f"{WAIT_MINUTES}분 후 다시 시도합니다...")
            time.sleep(WAIT_SECONDS)

if __name__ == "__main__":
    # 단일 실행 대신 연속 실행 함수를 호출
    run_continuous_analysis()