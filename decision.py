from typing import TypedDict, Annotated, Sequence
from datetime import datetime, timedelta
import json
import os
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
from langsmith import Client
import langsmith
from dotenv import load_dotenv
from database_manager import DatabaseManager
from price_collector import BithumbTrader
from news_collector import NaverNewsCollector
from trading import BithumbTradeExecutor
import time

# 환경 변수 및 LangSmith 설정
load_dotenv()
try:
    INVESTMENT = float(os.getenv('INVESTMENT'))
except TypeError:
    raise ValueError("INVESTMENT 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "crypto-trading-analysis"


# 상태 타입 정의
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next_step: str
    results: dict
    market_data: dict

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
    """시장 데이터를 매번 새로 수집하여 state에 저장"""
    try:
        # 항상 새로운 데이터를 수집
        if not trader.analyzer.data_queue.empty():
            market_data = trader.analyzer.data_queue.get()
        else:
            market_data = trader.collect_market_data()
        
        state['market_data'] = market_data
        
    except Exception as e:
        print(f"시장 데이터 수집 중 오류 발생: {e}")
        state['market_data'] = None
    
    return state['market_data']

@tool
def get_recent_news(dummy: str = "") -> str:
    """최근 24시간 동안의 암호화폐 관련 뉴스를 가져옵니다."""
    with langsmith.trace(
        name="get_recent_news",
        project_name="crypto-trading-analysis",
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
        project_name="crypto-trading-analysis",
        tags=["news", "analysis"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        news_data = get_recent_news.invoke("")
        
        prompt = f"""당신은 암호화폐 뉴스 분석 전문가입니다. 
        최근 24시간 동안의 암호화폐 관련 뉴스만을 분석하여 투자 결정을 내려주세요.
        기술적 지표나 가격은 고려하지 말고, 순수하게 뉴스 내용만으로 판단해주세요.
        
        뉴스 데이터:
        {news_data}
        
        다음 형식으로 분석 결과를 제공해주세요:
        1. 투자 결정: (매수/매도/관망)
        2. 투자 비중: (0-100%)
        3. 결정 이유: (뉴스 기반 분석)
        4. 주요 뉴스 요약
        5. 시장 영향도: (상/중/하)
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
        project_name="crypto-trading-analysis",
        tags=["price", "analysis"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        market_data = get_market_data_once(state)
        
        prompt = f"""당신은 암호화폐 기술적 분석 전문가입니다. 
        다음 데이터를 기반으로 투자 결정을 내려주세요.

        30분 기준 기술적 지표:
        - RSI: {market_data['analysis']['30m']['rsi']}
        - Stochastic K/D: {market_data['analysis']['30m']['stochastic'][0]}/{market_data['analysis']['30m']['stochastic'][1]}
        - MACD: {market_data['analysis']['30m']['macd'][0]}
        - 볼린저 밴드: 
          상단: {market_data['analysis']['30m']['bollinger_bands'][0]}
          중간: {market_data['analysis']['30m']['bollinger_bands'][1]}
          하단: {market_data['analysis']['30m']['bollinger_bands'][2]}
        
        24시간 추세:
        - 이동평균선: {market_data['analysis']['24h']['moving_averages']}
        - 거래량: {market_data['analysis']['24h']['ohlcv'][5]}
        
        호가 데이터:
        - 매수호가: {market_data['orderbook']['bids'][:5]}
        - 매도호가: {market_data['orderbook']['asks'][:5]}
        
        다음 형식으로 분석 결과를 제공해주세요:
        1. 투자 결정: (매수/매도/관망)
        2. 투자 비중: (0-100%)
        3. 기술적 분석 요약
        4. 주요 지표 해석
        5. 목표가: (매수/매도 시)
        6. 손절가: (매수/매도 시)
        7. 투자 시점: (단기/중기/장기)
        """
        
        response = llm.invoke(prompt)
        timestamp = datetime.now()
        current_price = market_data['current_price']['closing_price']
        
        state['results']['price_analysis'] = {
            'analysis': response.content,
            'timestamp': timestamp.isoformat()
        }
        
        db_manager.save_price_analysis(
            timestamp=timestamp,
            current_price=current_price,
            analysis_text=response.content
        )
        
        return state

def final_decision_agent(state: AgentState) -> AgentState:
    """뉴스 분석과 기술적 분석을 종합하여 최종 투자 결정을 내리는 에이전트"""
    with langsmith.trace(
        name="final_decision_agent",
        project_name="crypto-trading-analysis",
        tags=["final", "decision"]
    ):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        market_data = get_market_data_once(state)
        current_price = market_data['current_price']['closing_price']
        
        prompt = f"""당신은 암호화폐 투자 최고 결정권자입니다.
        뉴스 분석과 기술적 분석 결과를 종합하여 최종 투자 결정을 내려주세요.
        결정을 내릴때 뉴스 7, 가격 3의 비율로 생각해서 결정해주세요. 즉, 뉴스의 영향력이 더 큽니다.
        
        뉴스 분석 결과:
        {state['results']['news_analysis']['analysis']}
        
        기술적 분석 결과:
        {state['results']['price_analysis']['analysis']}
        
        현재 시장 상황:
        - 현재가: {current_price}
        - RSI: {market_data['analysis']['30m']['rsi']}
        - MACD: {market_data['analysis']['30m']['macd'][0]}
        
        다음 형식으로 최종 결정을 제공해주세요:
        1. 최종 투자 결정: (매수/매도/관망)
        2. 최종 투자 비중: (0-100%)
        3. 결정 이유: (종합적인 분석)
        4. 위험도: (상/중/하)
        5. 투자 전략: (단기/중기/장기)
        6. 목표가 및 손절가
        7. 주의사항
        """
        
        response = llm.invoke(prompt)
        timestamp = datetime.now()
        
        state['results']['final_decision'] = {
            'decision': response.content,
            'timestamp': timestamp.isoformat()
        }
        
        db_manager.save_final_decision(
            timestamp=timestamp,
            current_price=current_price,
            analysis_text=response.content
        )
        
        return state

def execute_trading_decision(state: AgentState) -> None:
    try:
        if 'final_decision' not in state['results']:
            print("최종 결정이 없어 거래를 실행할 수 없습니다.")
            return
        
        final_decision = state['results']['final_decision']
        current_price = float(state['market_data']['current_price']['closing_price'])
        timestamp = datetime.now()
        
        # 최종 결정 텍스트에서 거래 유형 파싱
        decision_text = final_decision['decision'].lower()
        
        # 관망 결정인 경우
        if "관망" in decision_text:
            print("\n=== 거래 실행 결과 ===")
            print("관망 결정으로 인해 거래를 실행하지 않습니다.")
            
            hold_order_id = f"HOLD_{timestamp.strftime('%Y%m%d%H%M%S')}"
            db_manager.save_trade_execution(
                timestamp=timestamp,
                trade_type="HOLD",  # 순수한 관망 결정
                quantity=0,
                price=current_price,
                total_amount=0,
                order_id=hold_order_id
            )
            return

        # 매수 결정일 때 잔액 확인
        if "매수" in decision_text:
            try:
                # Bithumb API로부터 잔액 정보 가져오기
                trade_executor = BithumbTradeExecutor()
                balance_info = trade_executor.get_balance()  # 잔액 조회 메서드 필요
                available_krw = float(balance_info.get('available_krw', 0))
                
                # 잔액이 만원 미만이면 BUY_CANT로 처리
                if available_krw < 10000:
                    print("\n=== 거래 실행 결과 ===")
                    print(f"잔액({available_krw:,.0f}원)이 만원 미만이라 매수할 수 없습니다.")
                    
                    buy_cant_order_id = f"BUY_CANT_{timestamp.strftime('%Y%m%d%H%M%S')}"
                    db_manager.save_trade_execution(
                        timestamp=timestamp,
                        trade_type="CANT_BUY",  # 잔액 부족으로 매수 불가
                        quantity=0,
                        price=current_price,
                        total_amount=0,
                        order_id=buy_cant_order_id
                    )
                    return
                    
            except Exception as e:
                print(f"잔액 확인 중 오류 발생: {e}")
                return

        # 거래 실행기 초기화 (이미 초기화되어 있지 않은 경우)
        try:
            if not trade_executor:
                trade_executor = BithumbTradeExecutor()
        except ValueError as ve:
            print(f"거래 실행기 초기화 실패: {ve}")
            return

        # 거래 실행
        result = trade_executor.execute_trade(
            decision=final_decision,
            max_investment=INVESTMENT,
            current_price=current_price
        )
        
        # 거래 결과 출력 및 저장
        print("\n=== 거래 실행 결과 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 거래 결과를 데이터베이스에 저장 (성공한 경우에만)
        if result['status'] == 'SUCCESS':
            db_manager.save_trade_execution(
                timestamp=timestamp,
                trade_type=result['type'],
                quantity=result['quantity'],
                price=result['price'],
                total_amount=result['total_amount'],
                order_id=result['order_id']
            )
    except Exception as e:
        print(f"거래 실행 중 오류 발생: {e}")

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
            project_name="crypto-trading-analysis",
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
                    print(result['results']['final_decision']['decision'])
                    
                    # 거래 실행 추가
                    execute_trading_decision(result)
            else:
                print("분석 결과가 없습니다.")
        
    except Exception as e:
        print(f"분석 실행 중 오류 발생: {e}")
        raise e

def run_continuous_analysis():
    """30분마다 트레이딩 분석을 실행하는 연속 실행 함수"""
    WAIT_MINUTES = 20
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