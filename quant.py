import finterstellar as fs
import matplotlib.pyplot as plt

graph_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\graphs\quant"

symbols = ['AAPL', 'MSFT', 'T', 'NVDA', 'GOOGL'] # 종목 리스트
cost = 0.001 # 매매 수수료
rf_rate = 0.0175 # 무위험 이자율

print("\n_______________________________________\n")

for symbol in symbols: # 각 종목에 대해 백테스팅
    df = fs.get_price(symbol, start_date='2022-01-01', end_date='2023-12-31') # 날짜 범위에 대한 종목의 주가 가져오기
    #print(df.tail())
    
    hp = df[symbol].max() # 최고가
    lp = df[symbol].min() # 최저가
    dif_ratio = (hp - lp) / lp # 최고가와 최저가 차이 비율
    s = 0.033 + 0.015 * dif_ratio # w값 종목별로 유동적으로 수정
    fs.envelope(df, w=30, spread=s) # 엔벨로프 계산
    #print(df.tail())
    #fs.draw_band_chart(df) # 엔벨로프 그래프 생성
    #plt.title(f"{symbol} Envelpe Graph (s = {s})")
    #plt.savefig(graph_path + '/' +  f"{symbol}_envelpoe_{s:.4f}.png") # 그래프 저장
    #print(f"Saving {symbol}_{s:.4f} Succesfully")
    #plt.show()
    
    print(f"{symbol} 모멘텀 전략 결과:")
    fs.band_to_signal(df, buy='A', sell='B') # 모멘텀 시그널
    fs.position(df)
    fs.evaluate(df, cost=cost)
    fs.performance(df, rf_rate=rf_rate) # 백테스팅
    print("\n_______________________________________\n")
    
    print(f"{symbol} 평균회귀 전략 결과:")
    fs.band_to_signal(df, buy='D', sell='B') # 평균회귀 시그널
    fs.position(df)
    fs.evaluate(df, cost=cost)
    fs.performance(df, rf_rate=rf_rate) # 백테스팅
    print("\n_______________________________________\n")