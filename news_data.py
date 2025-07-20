import json
import os
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


tickers = ['T', 'NVDA', 'GOOGL']
names = ['테슬라', '엔비디아', '구글']

data_dir = r'C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--log-level=3")

driver = webdriver.Chrome(options=chrome_options)

def news_data_collect(query, start_date, end_date, max_news=10):
    url = f"https://search.naver.com/search.naver?where=news&query={query}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={start_date}&de={end_date}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom{start_date}to{end_date}&is_sug_officeid=0&office_category=0&service_area=0"
    print(f"뉴스데이터 수집: {query} - {start_date} ~ {end_date}")
    
    driver.get(url)
    time.sleep(2)
    
    headlines = []
    elements = driver.find_elements(By.CSS_SELECTOR, "a.news_tit")
    for element in elements[:max_news]:
        title = element.get_attribute('title')
        clean_title = BeautifulSoup(title, 'html.parser').get_text().replace('\\', '')
        headlines.append(clean_title)
    
    return headlines

def fetch_news(ticker, fullname, start_date, end_date):
    news_data = {}
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + timedelta(weeks=1)
        start_date = current_date.strftime('%Y.%m.%d')
        end_date = (next_date - timedelta(days=1)).strftime('%Y.%m.%d')
        weekly_headlines = news_data_collect(fullname, start_date, end_date)
        news_data[current_date.strftime("%Y-%m-%d")] = weekly_headlines
        current_date = next_date

    with open(os.path.join(data_dir, f'news_data_{ticker}.json'), 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)

    return news_data

all_news_data = {}
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 12, 31)

for ticker, fullname in zip(tickers, names):
    all_news_data[ticker] = fetch_news(ticker, fullname, start_date, end_date)

driver.quit()