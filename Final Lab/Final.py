import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
csv_name = r"c:\Users\Jimmy\Desktop\Recommender_Systems\Recommended_system_Final\Final Lab\\1. train.csv"
df = pd.read_csv(csv_name)
#先選會用到的
# df = df[['ID','縣市','鄉鎮市區','路名','土地面積','建物型態','主要用途','主要建材','屋齡','建物面積','橫坐標','縱坐標','主建物面積','陽台面積','附屬建物面積','單價']]
df = df[['ID','縣市','建物型態','主要用途','單價']]

#設定一次就行的部分
# 該改為支持 中文的字體 
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决負號顯示
plt.figure(figsize=(18, 6))  # 將圖形寬度設置為18，高度設置為6
y_max = df['單價'].max() # 設定刻度的上限

# Q1 
# 將相同縣市名稱的索引設置為相同的值
# df['縣市索引'] = df.groupby('縣市').ngroup()
# img_price_city = df[['縣市', '單價', '縣市索引']]
# idx_unique = img_price_city['縣市索引'].unique()
# 顯示中文
idx_unique = df['縣市'].unique()
plt.subplot(1, 3, 1)                 # plt.subplot(列數, 行數, 圖形編號)設定第1
# plt.scatter(img_price_city['縣市索引'], img_price_city['單價'])
plt.scatter(df['縣市'], df['單價'])
plt.title('The relationship between housing price and city')
plt.xlabel('city')
plt.ylabel('Housing price')
# 設置 x 軸刻度位置和標籤
y_ticks = np.arange(0, y_max , 2.5)
plt.xticks(range(len(idx_unique)), idx_unique, rotation=45 , fontsize=12)
plt.yticks(y_ticks)
plt.grid(ls='--', lw=0.5, c='gray', axis='y')

# # Q2 
# df['用途索引'] = df.groupby('主要用途').ngroup()
# img_price_mainPurpose = df[['主要用途', '單價', '用途索引']]
# img_price_mainPurpose = img_price_mainPurpose.sort_values(by=['主要用途'])
# idx_unique = img_price_mainPurpose['用途索引'].unique()
idx_unique = df['主要用途'].unique()
plt.subplot(1, 3, 2)                 # plt.subplot(列數, 行數, 圖形編號)設定第2
# plt.scatter(img_price_mainPurpose['用途索引'], img_price_mainPurpose['單價'])
plt.scatter(df['主要用途'], df['單價'])
plt.title('The relationship between housing price and main purpose')
plt.xlabel('main purpose')
plt.ylabel('Housing price')
y_ticks = np.arange(0, y_max , 2.5)
plt.xticks(range(len(idx_unique)), idx_unique, rotation=45 , fontsize=12)
plt.yticks(y_ticks)
# axis：画哪个轴的网格线，默认x轴和y轴都画 c : 顏色
plt.grid(ls='--', lw=0.5, c='gray', axis='y')

# Q3 
# df['建物型態索引'] = df.groupby('建物型態').ngroup()
# img_price_buildingType = df[['建物型態', '單價', '建物型態索引']]
# img_price_buildingType = img_price_buildingType.sort_values(by=['建物型態'])
# idx_unique = img_price_buildingType['建物型態索引'].unique()
idx_unique = df['建物型態'].unique()
plt.subplot(1, 3, 3)                 # plt.subplot(列數, 行數, 圖形編號)設定第3
# plt.scatter(img_price_buildingType['建物型態索引'], img_price_buildingType['單價'])
plt.scatter(df['建物型態'], df['單價'])
plt.title('The relationship between housing price and building type')
plt.xlabel('building type')
plt.ylabel('Housing price')
y_ticks = np.arange(0, y_max , 2.5)
plt.xticks(range(len(idx_unique)), idx_unique, rotation=45 , fontsize=12)
plt.yticks(y_ticks)
# axis：画哪个轴的网格线，默认x轴和y轴都画 c : 顏色
plt.grid(ls='--', lw=0.5, c='gray', axis='y')

# 顯示圖形
plt.tight_layout()  # 自動調整子圖參數以適應圖形區域
plt.show()