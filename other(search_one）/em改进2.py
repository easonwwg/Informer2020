# 经过测试 k=10的时候 分类效果指标好
from decimal import Decimal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsClassifier
import impyute as impy
from ycimpute.imputer import EM

data = pd.read_csv('total.csv')
date = data.loc[:, 'date']
fs = data.loc[:, 'fs']
llgl = data.loc[:, 'llgl']
gl = data.loc[:, 'gl']
# 截取指定的列
totaldata = data.loc[:, ['fs', 'gl']]
print("总长度：" + str(len(totaldata)))
fullData = totaldata[(totaldata['gl'] >= 0)].copy()
print("不缺失长度：" + str(len(fullData)))
inputationData = totaldata[(totaldata['gl'] < 0)].copy()
print("缺失长度：" + str(len(inputationData)))

# 将值单位转换 转为mw  ，现在都是kw
fullData['gl'] = fullData['gl'] / 1000
# fullData['llgl'] = fullData['llgl'] / 1000
inputationData['gl'] = inputationData['gl'] / 1000
# inputationData['llgl'] = inputationData['llgl'] / 1000

# 不缺失的聚类
std = StandardScaler()  # 标准化 mean和std回受异常值影响，要进行异常值处理，转换后范围[-1,1],常用于聚类中
fullData = std.fit_transform(fullData)  # 先标准化
n_clusters = 10
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(fullData)
centers = cluster.cluster_centers_  # 簇心
centers = std.inverse_transform(centers)  # 再将簇心转为原始坐标，反标准化
fullData = std.inverse_transform(fullData)
print(centers)
labels = cluster.labels_  # 每一行的标签

##还原成带lable数据 这里用不到
list = fullData.tolist()
for i in range(0, len(list)):
    list[i].append(llgl[i] / 1000)
withllgldata = np.array(list)

# 在训练数据集中找到与新数据最邻近的k个实例，如果这k个实例的多数属于某个类别，那么新数据就属于这个类别。
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(fullData, labels)

# 创建字典
# dic = {}
# for j in range(0, len(labels)):
#     currenttype = labels[j]  # 每个lable
#     if currenttype in dic.keys():
#         pre = dic[currenttype]
#         pre.append(fullData[j])
#         dic[currenttype] = pre
#     else:
#         dic[currenttype] = [fullData[j]]

# 在fulldata添加类别

print('\r\n')
# 对缺失的数据填充
point = []
withllgllist = withllgldata.tolist()
for i in range(0, len(inputationData)):  # i是索引 x是值
    item = inputationData.values[i]
    fs = item[0]
    # 找到风速对应的理论功率
    cullgl = llgl[inputationData.index[i]] / 1000
    x = [[fs, cullgl]]
    predicttype = knn.predict(x)  # 根据风速和理论功率塞到风速和功率的模型中去预测
    dis, indexs = knn.kneighbors(x, 100)  # 查找点的K个邻居。返回距离和位置
    for index, va in enumerate(indexs[0]):
        point.append(withllgldata[va])  # 需不需要改成带理论功率的withllgldata，做到em再思考
    print('此次预测的类别是' + str(predicttype[0]))
    # print("根据point计算em的均值是和方差是")
    # 先不带初始化均值和方差计算
    # 开始进行em迭代，求预测功率值
    mean=np.array(point)[:,2].mean()
    std=np.array(point)[:,2].std()
    point.append([fs, cullgl, np.NaN])

    #第一个包预测
    # data_missing = pd.DataFrame(point)
    # em_imputed = impy.em(data_missing.values,mean,std)
    # print("预测的结果是：" + str(em_imputed[len(em_imputed) - 1]))

    fill = EM().complete(np.array(point),round(mean),round(std))
    #print("预测的结果是：" + str(fill[len(fill) - 1]))


    #del withllgllist[-1]  # 删除掉最后一条数据
    point = []
    print("\r\n")

# 再根据邻居算std和mean作为em的参数

# for j ,x in range(0, len(item)):
#     print(j)

a = 1
