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
from  sklearn.metrics import  calinski_harabasz_score

data = pd.read_csv('total.csv')
date = data.loc[:, 'date']
fs = data.loc[:, 'fs']
llgl = data.loc[:, 'llgl']
gl = data.loc[:, 'gl']
# 截取指定的列
nodateData = data.loc[:, ['fs', 'llgl', 'gl']]
print("去掉异常数值之前的长度：" + str(len(nodateData)))
nodateData = nodateData[(nodateData['gl'] > 0) & nodateData['llgl'] > 0]
print("去掉异常数值之后的长度：" + str(len(nodateData)))

# 将值单位转换 转为mw  ，现在都是kw
nodateData['gl'] = nodateData['gl'] / 1000
nodateData['llgl'] = nodateData['llgl'] / 1000

std = StandardScaler()  # 标准化 mean和std回受异常值影响，要进行异常值处理，转换后范围[-1,1],常用于聚类中
min_max_scaler = preprocessing.MinMaxScaler()  # 区间缩放-调整尺度 也就是归一化 同上 范围是[0,1]
normal = Normalizer() #正则化
nodateData = std.fit_transform(nodateData)

#记录轮廓系数
silhouettescore=[]
calinskiharabaszscoreList=[]
for i in range(5, 500):  # i是索引 x是值
    filename=open('image/1-500(inertia) .txt', 'a')
    n_clusters = i
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(nodateData)

    centroid = cluster.cluster_centers_  # 簇心
    # print(centroid)
    #将簇心反归一化
    centroid=std.inverse_transform(centroid)

    labels = cluster.labels_  # 每一行的label
    # print(labels)

    ##评价指标1 重要属性inertia_，查看总距离平方和，越小，模型效果越好。
    inertia = cluster.inertia_
    inertialog="k=" + str(i) + "时inertia_指标是：" + str(inertia)
    print(inertialog)
    filename.write(inertialog)
    filename.write('\n')

## 评价指标2 轮廓系数进行k值选择,曲线斜率变化快的部分就是分类的最佳选择
    #score=silhouette_score(nodateData,labels)
    #silhouettescore.append(score)

    ## 评价指标3 Calinski-Harabase 越大越好
    calinskiharabaszscore=calinski_harabasz_score(nodateData,labels)
    calinskiharabaszscoreList.append(calinskiharabaszscore)
    Calinskilog="k=" + str(i) + "时Calinski-Harabase指标是：" + str(calinskiharabaszscore)
    print(Calinskilog)
    filename.write(Calinskilog)
    filename.write('\r\n')
    filename.close()



plt.figure(figsize=(15,8))
plt.plot(range(5,500),calinskiharabaszscoreList,linewidth=1.5,linestyle='-')
plt.title("Calinski-Harabase")
plt.show()

#plt.figure(figsize=(10.6))
#plt.plot(range(210,500),silhouettescore,linewidth=1.5,linestyle='-')

# 开始聚类
