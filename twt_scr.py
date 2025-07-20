from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import datetime
import time
import json
import random
import urllib.parse 


USER_ID = "##########"
USER_PW = "################"
QUERY = ["Tesla", "Google", "Nvidia", "Apple", "Microsoft"]
START_DATE = "2025-07-01" 
END_DATE = "2025-07-16" 


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
]

options = Options()
options.add_argument('--log-level=1') 
#options.add_argument("--headless")
options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=options)


def twitter_login():
    try:
        driver.get("https://x.com/i/flow/login")
        print("트위터 로그인 페이지 로드 완료")
        
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.send_keys(USER_ID)
        username_input.send_keys(Keys.RETURN)
        time.sleep(2)
        
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )
        password_input.send_keys(USER_PW)
        password_input.send_keys(Keys.RETURN)
        time.sleep(5)
        print("로그인 성공")
        return True
    except Exception as e:
        print(f"로그인 실패: {str(e)}")
        return False


def scrape_tweets(query, date):
    url_query = urllib.parse.quote(
        f'"{query}" (news OR new OR breaking) since:{pre_date} until:{current_date_str} -filter:replies',
        safe=''
    )
    url = f"https://x.com/search?q={url_query}&src=typed_query&f=top"
    
    driver.get(url)
    print(f"검색 URL 로드 완료: {url}")
    
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
    )
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
    )
    
    tweets = []
    tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
    
    for tweet in tweet_elements[:50]:
        try:
            tweet_text = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
            tweets.append(tweet_text)
        except Exception as e:
                print(f"트윗 파싱 오류: {str(e)}")
    
    return tweets

results = {
    "search_data": []
}

if twitter_login():
    start = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.datetime.strptime(END_DATE, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]
    
    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n\n[ {date_str} ] 날짜 검색 시작")
        
        for query in QUERY:
            print(f"> 키워드 검색: '{query}'")
            try:
                tweets = scrape_tweets(query, date_str)
                results["search_data"].append({
                    "date": date_str,
                    "keyword": query,
                    "tweets": tweets
                })
                print(f"  - 수집된 트윗: {len(tweets)}개")
                print(tweets[0])
                time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(f"  ! {query} 검색 실패: {str(e)}")
    
    output_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\LSTM과 뉴스 헤드라인 감성분석을 통한 주식 예측\twt.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n모든 데이터 수집 완료: {output_path}")
else:
    print("로그인 실실패")
driver.quit()
