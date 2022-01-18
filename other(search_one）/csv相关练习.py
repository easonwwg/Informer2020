import random
import pandas as pd
import numpy as np

# 1 构建缺失值文件
data = pd.read_csv('total.csv')
date = data.loc[:, 'date']
fs = data.loc[:, 'fs']
llgl = data.loc[:, 'llgl']
gl = data.loc[:, 'gl']

rowsCount = len(date)
# 5%缺失的
fivePercentCount = round(rowsCount * 0.05)
fivePercentgl = gl.copy()  # 复制一个数组
# data = data.replace(' ', np.NaN) 将数组中某个值替换为nan
randowNum = sorted(
    random.sample(range(1, rowsCount), fivePercentCount))  # 从所有的rowsCount中生成指定fivePercentCount个数的不重复的数组,再排序
for i, x in enumerate(fivePercentgl):  # i是索引 x是值
    if ((i + 1) in randowNum):
        fivePercentgl[i] = np.NAN
print("处理后，数组中NAN的个数是" + str(len(fivePercentgl[np.isnan(fivePercentgl)])))

# 再将date fs llgl fivePercentgl 写入到csv中
a = np.column_stack(
    [date, fs, llgl, fivePercentgl])  # 拼接数组列组合 https://blog.csdn.net/qq_39516859/article/details/80666070
pd.DataFrame(a).to_csv("5%.csv", index=False, header=['date', 'fs', 'llgl', 'gl'])  # 写入文件

# 2 读取最新的含缺失值的文件，如果功率为缺失值，用我们的算法进行填充，并且和之前的数据进行对比


fifteenPercent = round(rowsCount * 0.15)
twentyPercent = round(rowsCount * 0.2)

## np.nan   判定缺失值 np.isnan(np.nan)
## count = np.array(data['gl'].astype(str).astype(float))   ##这个是转为一维数组
