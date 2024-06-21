import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 讀取訓練資料
train_data = pd.read_csv(r'1. train.csv')

# 定義特徵和目標變數
X = train_data[['縣市','建物型態','屋齡','車位個數','主要用途']]
y = train_data['單價']

# 處理類別變數
categorical_features = ['縣市','建物型態','主要用途']
numerical_features = ['車位個數','屋齡']

# 創建預處理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 建立隨機森林回歸模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 使用5-fold交叉驗證來訓練模型
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print(f'Cross-Validation RMSE Scores: {cv_rmse_scores}')
print(f'Average Cross-Validation RMSE: {cv_rmse_scores.mean()}')

# 訓練最終模型（使用所有訓練數據）
model.fit(X, y)

# 讀取測試資料
test_data = pd.read_csv(r'2. test.csv')

# 預測測試資料中的房價
X_test = test_data[['縣市','建物型態','屋齡','車位個數','主要用途']]
test_data['單價'] = model.predict(X_test)

# 保存預測結果
test_data.to_csv('test_predictions.csv', index=False)

print('Predictions saved to test_predictions.csv')