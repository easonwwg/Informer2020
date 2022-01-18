import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
#from scipy.interpolate import spline
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#mydata = pd.read_csv("缺失5%.csv")
# mydata = pd.read_csv("缺失10%.csv")
mydata = pd.read_csv("1.csv")
mydata1 = pd.read_csv("25%/缺失真实的数据.csv")
mydata2 = pd.read_csv("25%/缺失25%填充.csv")
count = np.array(mydata['gl'].astype(str).astype(float))
date = np.array(mydata['date'].astype(str))

count1 = np.array(mydata1['gl'].astype(str).astype(float))
date1 = np.array(mydata1['date'].astype(str))

count2 = np.array(mydata2['gl'].astype(str).astype(float))
date2 = np.array(mydata2['date'].astype(str))

xs = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in date]
fig = plt.figure(figsize=(8,6))#figsize : 指定figure的宽和高，单位为英寸
ax1 = fig.add_subplot(111)#参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块，如下图：
ax1.set_yticks(np.arange(0,2000,100))

xs1 = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in date1]

xs2 = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in date2]



plt.title('7/23-7/25完整数据集下功率曲线',fontsize=25)  # 字体大小设置为25
plt.xlabel('日期',fontsize=10)   # x轴显示“日期”，字体大小设置为10
plt.ylabel('负荷',fontsize=10)  # y轴显示“人数”，字体大小设置为10
plt.plot(xs, count,label='功率曲线')
# plt.plot(xs1, count1,label='缺失填充')
# plt.scatter(xs1, count1,marker='x',s=0.1,c='r',label='缺失填充')
# plt.scatter(xs2, count2,marker='x',s=0.1,c='g',label='填充真实数据')
ax1.scatter(xs1,count1,c='r',marker='x',label='mice')
ax1.scatter(xs2,count2,c='g',marker='x',label='缺失点原始真实值')

plt.xticks(rotation=45)
#plt.tick_params(axis='both',which='both',labelsize=10)

# 显示折线图
plt.legend()
plt.show()


## 1
# KNN
# em缺省值
# IterForest  随机森林
# 回归填充

##消融实验


# http://blog.sina.com.cn/s/blog_7fb03f7d01012j6p.html