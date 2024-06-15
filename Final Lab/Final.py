import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
csv_name = r"c:\Users\Jimmy\Desktop\Recommender_Systems\Recommended_system_Final\Final Lab\\1. train.csv"
df = pd.read_csv(csv_name)

df = df[['ID','縣市','鄉鎮市區','路名','土地面積','主要用途','主要建材','屋齡','建物面積','橫坐標','縱坐標','主建物面積','陽台面積','附屬建物面積','單價']]

# 將相同縣市名稱的索引設置為相同的值
df['縣市索引'] = df.groupby('縣市').ngroup()
img_price_city = df[['縣市', '單價', '縣市索引']]
img_price_city = img_price_city.sort_values(by=['縣市'])

city_idx_unique = img_price_city['縣市索引'].unique()

plt.figure()
plt.scatter(img_price_city['縣市索引'], img_price_city['單價'])
plt.title('The relationship between housing price and city')
plt.xlabel('city')
plt.ylabel('price')
# 設置 x 軸刻度位置和標籤
y_min = df['單價'].min()
y_max = df['單價'].max()
y_ticks = np.arange(0, y_max , 2.5)
plt.xticks(range(len(city_idx_unique)), city_idx_unique)
plt.yticks(y_ticks)
# axis：画哪个轴的网格线，默认x轴和y轴都画 c : 顏色
plt.grid(ls='--', lw=0.5, c='gray', axis='y')

plt.show()