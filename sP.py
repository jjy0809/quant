import pandas as pd  # 데이터 조작을 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리
from keras.models import load_model  # 저장된 Keras 모델 로드
from sklearn.preprocessing import MinMaxScaler  # 데이터 스케일링을 위한 라이브러리
import os  # 운영 체제 인터페이스를 위한 라이브러리
import pathlib  # 경로 처리를 위한 라이브러리

# 경로 설정
data_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data')
model_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\models')
graph_dir = pathlib.Path(r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\graphs')

# 그래프 디렉토리가 존재하지 않으면 생성
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

# 주식 티커 리스트
tickers = [ 'MSFT']

# 주식 데이터 불러오기
stock_data = {ticker: pd.read_csv(data_dir / f'stock_data_{ticker}.csv', index_col=0, parse_dates=True) for ticker in tickers}

# 주식 데이터를 전처리하는 함수
def preprocess_stock_data(data):
    scaler = MinMaxScaler()  # MinMaxScaler 초기화
    data['Change'] = data['Close'].pct_change()  # 일일 변동률 계산
    data['Volume_Change'] = data['Volume'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)  # NaN 값 제거
    features = ['Change', 'Volume_Change', 'MA_5', 'MA_20']
    scaled_data = scaler.fit_transform(data[features].values)  # 변동률 데이터를 스케일링
    return scaled_data, scaler

# 예측을 위한 데이터 준비
def prepare_prediction_data(model, stock_data, scaler, window_size=60, steps=52):
    stock_values = stock_data[-window_size:]  # 최근 stock 데이터
    
    X_test = np.array(stock_values).reshape(1, window_size, -1)  # (1, window_size, 4)
    predictions = []
    for i in range(steps):
        predicted_scaled = model.predict(X_test)
        
        # Prepare the full input shape with zero padding for inverse transformation
        predicted_scaled_full = np.zeros((predicted_scaled.shape[0], 4))
        predicted_scaled_full[:, 0] = predicted_scaled[:, 0]
        
        predicted_change = scaler.inverse_transform(predicted_scaled_full)[0, 0]  # 스케일링된 값을 원래 값으로 변환
        predictions.append(predicted_change)
        
        # 다음 윈도우 업데이트
        next_window = np.zeros((1, window_size, 4))
        next_window[0, :-1, :] = X_test[0, 1:, :]
        next_window[0, -1, :] = predicted_scaled_full[0, :]
        
        X_test = next_window
    return predictions

# 미래 주가 예측
def predict_future_prices(model, stock_data, scaler, window_size=60, steps=52):
    predictions = prepare_prediction_data(model, stock_data, scaler, window_size, steps)
    return predictions

# 예측된 변동률을 적용하여 주가를 계산하는 함수
def plot_predictions(stock_data, predicted_changes, ticker, path):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.loc['2022-01-01':'2023-12-31', 'Close'], label='Actual', color='blue')  # 기간 내 실제 주가 플롯
    last_price = stock_data['Close'].loc['2022-12-30']
    predicted_prices = [last_price * (1 + change) for change in np.cumsum(predicted_changes)]
    predicted_dates = pd.date_range(start='2023-01-01', periods=len(predicted_prices), freq='W-FRI')
    plt.plot(predicted_dates, predicted_prices, label='Predicted', linestyle='--', color='red')  # 예측된 주가 플롯
    plt.legend()
    plt.title(f"{ticker} Stock Prices and Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(path / f"{ticker}_predictions.png")  # 그래프 저장
    plt.show()  # 그래프 표시

# 수익률 계산
def calculate_returns(stock_data, predicted_changes):
    last_actual_price = stock_data['Close'].loc['2022-12-30']
    predicted_price = last_actual_price * (1 + np.sum(predicted_changes))
    returns = (predicted_price - last_actual_price) / last_actual_price * 100
    return returns

# 수익률 기준으로 주식 정렬
def sort_stocks_by_returns(stock_data, tickers, path):
    returns = {}
    for ticker in tickers:
        data = stock_data[ticker].copy()
        data['Change'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        scaled_data, scaler = preprocess_stock_data(data)  # 주식 데이터 전처리
        model_path = model_dir / f'model_{ticker}.keras'
        model = load_model(str(model_path))
        predicted_changes = predict_future_prices(model, scaled_data, scaler)
        returns[ticker] = calculate_returns(stock_data[ticker], predicted_changes)
        plot_predictions(stock_data[ticker], predicted_changes, ticker, path)
    sorted_returns = sorted(returns.items(), key=lambda item: item[1], reverse=True)
    return sorted_returns

# 메인 실행
sorted_stocks = sort_stocks_by_returns(stock_data, tickers, graph_dir)
print(sorted_stocks)
