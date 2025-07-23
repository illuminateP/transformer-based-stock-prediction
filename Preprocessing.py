import numpy as np
from scipy.stats import skew
from scipy.stats import entropy
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('삼성전자 주식데이터2.csv', encoding='cp949')
close = df['종가'].to_numpy(dtype=np.float64)
volumes = df['거래량'].to_numpy(dtype=np.float64)
open = df['시가'].to_numpy(dtype=np.float64)
high = df['고가'].to_numpy(dtype=np.float64)
low = df['저가'].to_numpy(dtype=np.float64)
market_cap = df['시가총액'].to_numpy(dtype=np.float64)
shares_outstanding = df['상장주식수'].to_numpy(dtype=np.float64)
up_down = df['결과'].to_numpy(dtype=np.float64)     # 예측 결과값 배열

# 변수 선언
width = 20  # 특성 계산을 위한 윈도우 크기 설정

result = []  # 결과를 저장할 리스트

for i in range(0, len(df)-width-1):
    close_width = close[i:i+width]  # 종가 데이터
    volumes_width = volumes[i:i+width]  # 거래량 데이터
    open_width = open[i:i+width]  # 시가 데이터
    high_width = high[i:i+width]  # 고가 데이터
    low_width = low[i:i+width]  # 저가 데이터
    market_cap_width = market_cap[i:i+width]  # 시가총액
    shares_outstanding_width = shares_outstanding[i:i+width]  # 상장주식수
    
    # 대비 (현재 종가 - 이전 종가)
    price_diff = close_width[-1] - close_width[-2]
    
    # 등락률 (현재 종가와 이전 종가 비율)
    price_change = (close_width[-1] / close_width[-2]) - 1
    
    result.append([
        # 5일 이동평균
        np.mean(close_width[-5:]),
        
        # 20일 이동평균
        np.mean(close_width),
        
        # 5일 표준편차
        np.std(close_width[-5:]),
        
        # 5일 이동평균 대비 현재 종가 비율
        close_width[-1] / np.mean(close_width[-5:]),
        
        # 5일 이동평균 대비 현재 거래량 비율
        volumes_width[-1] / np.mean(volumes_width[-5:]),
        
        # 볼린저 밴드 위치
        (close_width[-1] - (np.mean(close_width) - 2 * np.std(close_width))) / (4 * np.std(close_width)),
        
        # Z-score : 표준화 점수
        (close_width[-1] - np.mean(close_width[-5:])) / np.std(close_width[-5:]),
        
        # 가격 거래량 비율
        np.mean(np.abs(np.diff(close_width)[-5:])) / np.mean(volumes_width[-5:]),
        
        # 가격 가속도
        (close_width[-1] - close_width[-2]) - (close_width[-2] - close_width[-3]),
        
        # 추세 강도
        sum(1 for i in range(1, len(close_width)) if close_width[i] > close_width[i-1]) / (len(close_width)-1),
        
        # 시가 대비 종가 변화율
        (close_width[-1] - open_width[-1]) / open_width[-1],
        
        # 고가 대비 저가 변화율
        (high_width[-1] - low_width[-1]) / open_width[-1],
        
        # 시가 대비 종가 비율
        close_width[-1] / open_width[-1],
        
        # 고가 대비 저가 차이
        high_width[-1] - low_width[-1],
        
        # 시가총액 대비 거래대금 비율
        np.mean(volumes_width) / np.mean(market_cap_width),
        
        # 상장주식수 대비 거래량 비율
        np.mean(volumes_width) / np.mean(shares_outstanding_width),
        
        # 평균 거래량 대비 변화율
        (volumes_width[-1] - np.mean(volumes_width)) / np.mean(volumes_width),
        
        # 고가 대비 저가 비율
        low_width[-1] / high_width[-1],

        # 일별 종가 대비 시가 비율 (시가: open, 종가: close)
        close_width[-1] / open_width[-1],

        # 최근 5일 이동평균 대비 고가 비율
        high_width[-1] / np.mean(close_width[-5:]),
        
        # 최근 5일 이동평균 대비 저가 비율
        low_width[-1] / np.mean(close_width[-5:]),
        
        # 5일 거래량 변화율
        (volumes_width[-1] - volumes_width[-5]) / volumes_width[-5],
        
        # 일별 고가-저가 비율
        (high_width[-1] - low_width[-1]) / open_width[-1],
        
        # 5일 고가-저가 차이의 표준편차
        np.std(np.array(high_width[-5:]) - np.array(low_width[-5:])),
        
        # 추세 전환 점수 (예시: 5일 이동평균 크로스오버)
        np.mean(np.diff(close_width[-5:])) / np.mean(close_width[-5:]),
        
        # 가격 변동성 (5일)
        np.std(np.diff(close_width[-5:])),
        
        # 등락률에 따른 변화
        price_change,
        
        up_down[i+width]
    ])

    
df4_result = pd.DataFrame(result, columns=['5일 이동평균', '20일 이동평균', '5일 표준편차', '5일 이동평균 대비 현재 종가 비율',
                                           '5일 이동평균 대비 현재 거래량 비율', '볼린저 밴드 위치', 'Z-score', '가격 거래량 비율',
                                           '가격 가속도', '추세 강도', '시가 대비 종가 변화율', '고가 대비 저가 변화율',
                                           '시가 대비 종가 비율', '고가 대비 저가 차이', '시가총액 대비 거래 대금 비율', '상장주식수 대비 거래량 비율',
                                           '평균 거래량 대비 변화율', '고가 대비 저가 비율', '일별 종가 대비 시가 비율', '최근 5일 이동 평균 대비 고가 비율',
                                           '최근 5일 이동평균 대비 저가 비율', '5일 거래량 변화율', '일별 고가-저가 비율',
                                           '5일 고가 - 저가 차이의 표준편차', '추세 전환 점수', '가격 변동성', '등락률에 따른 변화',
                                           '예측 결과값'])

df4_result.to_csv('result_A.csv', index=False, header=False, encoding='cp949')

data = np.vstack(df4_result.values.astype(float))
np.random.seed(42)
np.random.shuffle(data)

total_rows = data.shape[0]
split_index = int(total_rows * 2 / 3)

traindata = data[:split_index, :]
testdata = data[split_index:, :]

np.savetxt('result_train.csv', traindata, delimiter=',', fmt='%f', encoding='utf-8')
np.savetxt('result_test.csv', testdata, delimiter=',', fmt='%f', encoding='utf-8')