

# 개발 환경
    - OS : Ubuntu 18.04.5 LTS
    - CPU : Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
    - GPU : Quadro RTX 8000




# 필요한 패키지

    - requirements.txt에 필요한 패키지와 각 라이브러리의 버전이 기록 되어 있다. 
    - 설치를 위해 인터프리터에서 현재 파일 경로로 이동 후 다음 코드를 실행해준다. 코드:pip install -r requirements.txt
        
        
        
        
# 실행순서 
    
    - 모든 실행 파일에 데이터경로 수정 필요함.

    - 모든 실행 파일은 .py파일로, 인터프리터를 통해 실행해야함. 
        - ex) python prediction.py

    - prediction.py 
        - 예측모델 학습 파일로, ./output/output_tabnet_sub/ 에 TEST_01.csv 부터 TEST_05.csv 까지 제출 파일이 생성된다.
        - 예측모델이 학습한 weight는 ./model/prediction/tabnet_prediction.zip으로 저장된다.
        
    - collect_good_case_train.py
        - 생성모델에 필요한 dataset을 만드는 파일
        
    - collect_good_case_test.py
        - 생성모델에 필요한 dataset을 만드는 파일
        
    - collect_good_case_per_week.py
        - 생성모델에 필요한 dataset을 만드는 파일
        
    - generation_train.py
        - 좋았던 train case에 대한 생육환경 생성모델 파일로, ./output/output_generation_train/gen_dataset_tmp_(epoch수).csv 로 생육환경이 생성된다.
        - 생성된 파일은 완벽한 생육환경이 아니고 전처리가 필요하다. 
        - 생성모델이 학습한 weight는 ./model/generation_train/epochN/ 경로에 컬럼별 critic_x, critic_z, encoder, decoder의 모델 weight이 저장된다.
        
    - generation_test.py
        - 좋았던 test case에 대한 생육환경 생성모델 파일로, ./output/output_generation_test/gen_dataset_tmp_(epoch수).csv 로 생육환경이 생성된다.
        - 생성된 파일은 완벽한 생육환경이 아니고 전처리가 필요하다. 
        - 생성모델이 학습한 weight는 ./model/generation_test/epochN/ 경로에 컬럼별 critic_x, critic_z, encoder, decoder의 모델 weight이 저장된다.
        
    - generation_week.py
        - train+test set 중 주차별로 좋았던 케이스를 합친 dataset에 대한 생육환경 생성모델 파일로, ./output/output_generation_week/gen_dataset_tmp_(epoch수).csv 로 생육환경이 생성된다.
        - 생성된 파일은 완벽한 생육환경이 아니고 전처리가 필요하다. 
        - 생성모델이 학습한 weight는 ./model/generation_week/epochN/ 경로에 컬럼별 critic_x, critic_z, encoder, decoder의 모델 weight이 저장된다.    
        
    - generation_final_sub.py
        - 생성모델 기반으로 나온 데이터셋을 pretrained 예측 모델에 넣어 최종 predicted_weight_g를 생성하여 생성모델의 최종 제출물을 만듦.
            - train case로부터 결과물은 ./output/from_train_case_generation_output_sub/generation.csv ,predicted_weight_g.csv 각각 x data, y data
            - week case로부터 결과물은 ./output/from_week_case_generation_output_sub/generation.csv ,predicted_weight_g.csv 각각 x data, y data
            - test case로부터 결과는 검증용이므로 submission에 포함하지 않음.
            
    
# 최적의 생육 환경 파일

    - ./output/from_train_case_generation_output_sub/  경로의 generation.csv, predicted_weight_g.csv
    - ./output/from_week_case_generation_output_sub/   경로의 generation.csv, predicted_weight_g.csv
        
        
        
읽어주셔서 감사합니다.
        
        
