import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


data_dir = r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data'
model_dir = r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 티커
tickers = ['MSFT'] #['AAPL']  # , 'MSFT', 'T', 'NVDA', 'GOOGL']

stock_data = {ticker: pd.read_csv(os.path.join(data_dir, f'stock_data_{ticker}.csv'), index_col=0, parse_dates=True) for ticker in tickers}

scaler = MinMaxScaler()

# 데이터 전처리
def preprocess(data):
    data['Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)
    features = ['Change', 'Volume_Change', 'MA_5', 'MA_20']
    scaled_data = scaler.fit_transform(data[features].values)
    return scaled_data, scaler

# 데이터 준비
def prepare_data(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y

# LSTM 모델 훈련
def train(X_train, y_train, X_val, y_val, save_path):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 조기 종료
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])
    model.save(save_path, save_format='keras')
    return model


# 종목별 학습습
for ticker in tickers:
    data = stock_data[ticker]
    data = data.loc[:'2022-12-31']
    data.dropna(inplace=True)
    scaled_data, scaler = preprocess(data)
    
    scaled_data = pd.DataFrame(scaled_data, index=data.index, columns=['Change', 'Volume_Change', 'MA_5', 'MA_20'])

    train_data = scaled_data[scaled_data.index < '2023-01-01']
    X, y = prepare_data(train_data.values)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    save_path = os.path.join(model_dir, f'model_{ticker}.keras')
    train(X_train, y_train, X_val, y_val, save_path)
    print(f"Save model {ticker}")
