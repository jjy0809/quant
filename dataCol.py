import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import time

# 주식 티커 및 회사명
tickers = ['AAPL', 'MSFT', 'T', 'NVDA', 'GOOGL']
names = ['애플', '마이크로소프트', '테슬라', '엔비디아', '구글']

# 데이터 디렉토리 생성
data_dir = r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 주식 데이터 다운로드 함수
def download_stock_data(ticker, max_retries=5):
    for attempt in range(max_retries):
        try:
            print(f"Getting stock price for {ticker} (Attempt {attempt + 1})...")
            data = yf.download(ticker, start='2018-01-01', end='2023-12-31')
            return data
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            time.sleep(0.5)  # 재시도 전 대기 시간
    return None

# 주식 데이터 다운로드
stock_data = {}
for ticker in tickers:
    data = download_stock_data(ticker)
    if data is not None:
        stock_data[ticker] = data
        stock_data[ticker].to_csv(os.path.join(data_dir, f'stock_data_{ticker}.csv'))


