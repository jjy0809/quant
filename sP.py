import pandas as pd  # 데이터 조작
import numpy as np  # 수학적 계산
import matplotlib.pyplot as plt  # 그래프 생성
from keras.models import load_model  # Keras 모델 로드
from sklearn.preprocessing import MinMaxScaler  # 데이터 스케일링
import os
import pathlib  # 경로 처리


data_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data')
model_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\models')
graph_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\graphs')

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

# 주식 종목 티커
tickers = [ 'MSFT']

stock_data = {ticker: pd.read_csv(data_dir / f'stock_data_{ticker}.csv', index_col=0, parse_dates=True) for ticker in tickers}

# 데이터 전처리리
def preprocess(data):
    scaler = MinMaxScaler()  
    data['Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)
    features = ['Change', 'Volume_Change', 'MA_5', 'MA_20']
    scaled_data = scaler.fit_transform(data[features].values)
    return scaled_data, scaler

# 데이터 준비 
def prepare_data(model, stock_data, scaler, window_size=60, steps=52):
    stock_values = stock_data[-window_size:]
    X_test = np.array(stock_values).reshape(1, window_size, -1)
    predictions = []
    for i in range(steps):
        predicted_scaled = model.predict(X_test)
        predicted_scaled_full = np.zeros((predicted_scaled.shape[0], 4))
        predicted_scaled_full[:, 0] = predicted_scaled[:, 0]
        predicted_change = scaler.inverse_transform(predicted_scaled_full)[0, 0]
        predictions.append(predicted_change)
        next_window = np.zeros((1, window_size, 4))
        next_window[0, :-1, :] = X_test[0, 1:, :]
        next_window[0, -1, :] = predicted_scaled_full[0, :]
        X_test = next_window
    return predictions

# 가격 예측 
def predict_price(model, stock_data, scaler, window_size=60, steps=52):
    pres = prepare_data(model, stock_data, scaler, window_size, steps)
    return pres

# 그래프 생성 
def graph_generate(stock_data, predicted_changes, ticker, path):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.loc['2022-01-01':'2023-12-31', 'Close'], label='Actual', color='blue')
    last_price = stock_data['Close'].loc['2022-12-30']
    predicted_prices = [last_price * (1 + change) for change in np.cumsum(predicted_changes)]
    predicted_dates = pd.date_range(start='2023-01-01', periods=len(predicted_prices), freq='W-FRI')
    plt.plot(predicted_dates, predicted_prices, label='Predicted', linestyle='--', color='red')
    plt.legend()
    plt.title(f"{ticker} Stock Prices and Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(path / f"{ticker}_predictions.png")
    plt.show()

# 수익 계산 
def calculate_profit(stock_data, predicted_changes):
    last_actual_price = stock_data['Close'].loc['2022-12-30']
    predicted_price = last_actual_price * (1 + np.sum(predicted_changes))
    returns = (predicted_price - last_actual_price) / last_actual_price * 100
    return returns

# 수익률 기준 내림차순 정렬렬
def sort_tickers(stock_data, tickers, path):
    returns = {}
    for ticker in tickers:
        data = stock_data[ticker].copy()
        data['Change'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        scaled_data, scaler = preprocess(data)
        model_path = model_dir / f'model_{ticker}.keras'
        model = load_model(str(model_path))
        predicted_changes = predict_price(model, scaled_data, scaler)
        returns[ticker] = calculate_profit(stock_data[ticker], predicted_changes)
        graph_generate(stock_data[ticker], predicted_changes, ticker, path)
    sorted_returns = sorted(returns.items(), key=lambda item: item[1], reverse=True)
    return sorted_returns


if __name__ == '__main__':
    sorted_stocks = sort_tickers(stock_data, tickers, graph_dir)
    print(sorted_stocks)
