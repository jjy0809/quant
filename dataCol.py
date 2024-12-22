import yfinance as yf
import os
import time

# 주식 티커 및 회사명
tickers = ['AAPL', 'MSFT', 'T', 'NVDA', 'GOOGL']
names = ['애플', '마이크로소프트', '테슬라', '엔비디아', '구글']

dir = r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data'
if not os.path.exists(dir):
    os.makedirs(dir)

# 주식 데이터 다운로드 함수
def stk_down(ticker, max=5):
    for i in range(max):
        try:
            print(f"Getting stock price for {ticker} (Attempt {i + 1})...")
            data = yf.download(ticker, start='2018-01-01', end='2023-12-31')
            return data
        except Exception as e:
            print(f"Download failled {ticker}: {e}")
            time.sleep(0.25)  
    return None

# 주식 데이터 다운로드
stk_data = {}
for ticker in tickers:
    data = stk_down(ticker)
    if data is not None:
        stk_data[ticker] = data
        stk_data[ticker].to_csv(os.path.join(dir, f'stock_data_{ticker}.csv'))


