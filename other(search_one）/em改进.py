# 经过测试 k=10的时候 分类效果指标好
import pandas as pd
import numpy as np
import sklearn.cluster as sc
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from decimal import Decimal

data = pd.read_csv('total.csv')
date = data.loc[:, 'date']
fs = data.loc[:, 'fs']
llgl = data.loc[:, 'llgl']
gl = data.loc[:, 'gl']
# 截取指定的列
totaldata = data.loc[:, ['fs', 'llgl', 'gl']]
print("总长度：" + str(len(totaldata)))
fullData = totaldata[(totaldata['gl'] >= 0)].copy()
print("不缺失长度：" + str(len(fullData)))
inputationData = totaldata[(totaldata['gl'] < 0)].copy()
print("缺失长度：" + str(len(inputationData)))

# 将值单位转换 转为mw  ，现在都是kw
fullData['gl'] = fullData['gl'] / 1000
fullData['llgl'] = fullData['llgl'] / 1000
inputationData['gl'] = inputationData['gl'] / 1000
inputationData['llgl'] = inputationData['llgl'] / 1000

# 不缺失的聚类
std = StandardScaler()  # 标准化 mean和std回受异常值影响，要进行异常值处理，转换后范围[-1,1],常用于聚类中
fullData = std.fit_transform(fullData)  # 先标准化
n_clusters = 10
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(fullData)
centers = cluster.cluster_centers_  # 簇心
centers = std.inverse_transform(centers)  # 再将簇心转为原始坐标，反标准化
print(centers)
labels = cluster.labels_  # 每一行的标签

print('\r\n')
# 对缺失的数据填充


a = 1
for i in range(0, len(inputationData)):  # i是索引 x是值
    item = inputationData.values[i]
    fs = Decimal(item[0]).quantize(Decimal("0.00"))
    llgl = Decimal(item[1]).quantize(Decimal("0.00"))
    gl = Decimal(item[2]).quantize(Decimal("0.00"))
    print(str(fs) + " " + str(llgl) + " " + str(gl))
    #根据 风速和理论功率找最近的簇心，
    for j in range(0,len(centers)):
        center_fs=cluster[j][0]
        centers_llgl=cluster[j][1]


    # 再根据最近的簇心找k个邻居，再根据邻居算std和mean作为em的参数

    # for j ,x in range(0, len(item)):
    #     print(j)

a = 1
