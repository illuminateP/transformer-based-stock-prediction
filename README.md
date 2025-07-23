# transformer-based-stock-prediction

## Overview  
삼성전자 일별 주가 데이터를 이용하여, 기초적인 feature engineering, 신경망(Multi-Layer Perceptron, Transformer) 기반 예측 모델을 비교·실험하는 Python 프로젝트.

## Tech Stack  
- **Language:** Python 3.12  
- **Libraries/Frameworks:**  
  - numpy, pandas, scikit-learn  
  - tensorflow / keras  
  - matplotlib  
  - scipy
  
## Description  
- 삼성전자 주가 데이터를 활용하여 다양한 특성(Feature)을 생성하고,  
- MLP와 Transformer 신경망 모델을 각각 학습/평가하여  
- 주가 등락(이진 분류) 예측 성능을 실험적으로 비교합니다.

코드는 재현성, 실험 로그, 결과 저장, 확장 실험(다른 종목, feature selection 등)에 초점을 맞춰 설계되어 있습니다.

## Directory  
- `/Preprocessing.py` : 데이터 전처리 코드  
- `/RNN.py` : RNN(순환 신경망) 실험 코드  
- `/Transformer.ipynb` : Transformer 실험 및 분석 노트북  
- `/result_A.csv`, `/result_test.csv`, `/result_train.csv` : 실험 결과(csv)  
- `/삼성전자 주식데이터2.csv` : 원본 주가 데이터  

## Installation & Usage  
1. Python 3.12 이상 환경을 준비하세요.  
2. 필수 패키지 설치:
   ```
   pip install -r requirements.txt
   ```
   
## Contact
문의: sincelife777@gmail.com (illuminateP)
