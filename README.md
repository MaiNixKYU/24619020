# 24619020
# 1. 드라이브 마운트 및 데이터 준비
from google.colab import drive
drive.mount('/content/drive')

!unzip --qq /content/drive/MyDrive/open.zip -d dataset

# 2. 라이브러리 설치
!pip install -q category_encoders catboost lightgbm imbalanced-learn

#  3. 라이브러리 임포트 및 데이터 로드
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import f1_score

from sklearn.preprocessing import LabelEncoder

from category_encoders import TargetEncoder

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTENC


train = pd.read_csv('dataset/train.csv')

test = pd.read_csv('dataset/test.csv')

X = train.drop(columns=['ID', 'Cancer'])

y = train['Cancer']

x_test = test.drop(columns=['ID'])

# 4. 범주형 변수 타겟 인코딩
categorical_features = [col for col in X.columns if X[col].dtype == 'object']

te = TargetEncoder()

X[categorical_features] = te.fit_transform(X[categorical_features], y)

x_test[categorical_features] = te.transform(x_test[categorical_features])

# 5. SMOTENC로 오버샘플링
cat_indices = [X.columns.get_loc(col) for col in categorical_features]

smote = SMOTENC(categorical_features=cat_indices, random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. 교차검증 및 앙상블 학습
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_preds = np.zeros(len(x_test))

f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):

    print(f"\n[Fold {fold+1}]")

    X_tr, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
    
    y_tr, y_val = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]

    # 개별 모델 정의
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    model_lgbm = LGBMClassifier(random_state=42)
    
    model_cat = CatBoostClassifier(verbose=0, random_state=42)

    # 앙상블 구성 (소프트 보팅)
    ensemble = VotingClassifier(estimators=[
    
        ('xgb', model_xgb),
        
        ('lgbm', model_lgbm),
        
        ('cat', model_cat)
        
    ], voting='soft', n_jobs=-1)

    # 학습
    ensemble.fit(X_tr, y_tr)

    # 검증
    val_pred = ensemble.predict(X_val)
    
    f1 = f1_score(y_val, val_pred)
    
    f1_scores.append(f1)
    
    print(f"F1 Score: {f1:.4f}")

    # 테스트 예측 (각 fold 예측 누적)
    final_preds += ensemble.predict_proba(x_test)[:, 1]

# 7. 최종 예측 및 제출 파일 생성
final_labels = (final_preds / skf.n_splits >= 0.8).astype(int)

submission = pd.read_csv('dataset/sample_submission.csv')

submission['Cancer'] = final_labels

submission.to_csv('ensemble_submit.csv', index=False)

from google.colab import files

files.download('ensemble_submit.csv')


