import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from dateutil import parser
from mpmath import arange

data = pd.read_csv('1.csv')
xdata = []
ydata = []
xdata = data.loc[:,'date']   #将csv中列名为“列名1”的列存入xdata数组中
#如果ix报错请将其改为loc
ydata = data.loc[:,'gl']   #将csv中列名为“列名2”的列存入ydata数组中

# plt.plot(xdata, xdata, color="red", linewidth=2)# Test case 3:
#
# plt.show()

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# x_data = ['2020/7/23 0:00','2020/7/23 0:15','2020/7/23 0:30','2020/7/23 0:45','2020/7/23 1:00','2020/7/23 1:15','2020/7/23 1:30']
x_data = ['2020/7/23','2020/7/24','2020/7/25','2020/7/26','2020/7/27','2020/7/28','2020/7/29']
y_data = [58000,60200,63000,71000,84000,90500,107000]
# xs = [datetime.strptime(d, '%Y/%m/%d').date() for d in x_data]
# plt.figure(1)
# plt.subplot(1, 3, 1)
plt.xticks(rotation=3)
plt.plot(x_data,y_data,color='red')
plt.show()

# plt.plot(xdata,ydata,'bo-',label=u'',linewidth=1)
# plt.title(u"表名",size=10)   #设置表名为“表名”
# plt.legend()
# plt.xlabel(u'x轴名',size=10)   #设置x轴名为“x轴名”
# plt.ylabel(u'y轴名',size=10)   #设置y轴名为“y轴名”
# plt.show()