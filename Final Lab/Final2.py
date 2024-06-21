import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# 讀取訓練資料
train_data = pd.read_csv(r"1. train.csv")
# 定義特徵
features  = ['縣市', '鄉鎮市區', '主要用途', '建物型態', '屋齡']
# 四舍五入 
train_data['屋齡'] = train_data['屋齡'].round() * 10

# 处理分类变量 ==> 文字轉換為數值分類
label_encoders = {}
for column in features:
    le = LabelEncoder() # 將文字 轉換為 數值
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le 

x_train = train_data[features]
y_train = train_data['單價']
# 創立決策樹模型 random_state 其中隨機種子固定 這樣每次預測結果都會是一樣的
model = DecisionTreeRegressor(random_state=42)

# 进行交叉验证
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
# 打印交叉验证结果
print("Cross-Validation Results for [cv = 5]:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score}")
# 计算平均得分
mean_score = cv_scores.mean()
print(f"\nAverage Score: {mean_score}")

# 訓練
model.fit(x_train, y_train)

test_data = pd.read_csv(r'2. test.csv')
# 四舍五入 
test_data['屋齡'] =test_data['屋齡'].round() * 10
# 处理分类变量 ==> 文字轉換為數值分類
for column in features:
    le = LabelEncoder() # 將文字 轉換為 數值
    test_data[column] = le.fit_transform(test_data[column])
    label_encoders[column] = le 
x_test = test_data[features]

test_data['單價'] = model.predict(x_test)
# 保存預測結果
test_data.to_csv('test_pred.csv', index=False)
