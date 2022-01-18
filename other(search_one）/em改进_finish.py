# 经过测试 k=10的时候 分类效果指标好
from decimal import Decimal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsClassifier
import impyute as impy

data = pd.read_csv('data(no need commit)\\total.csv')
date = data.loc[:, 'date']
fs = data.loc[:, 'fs']
llgl = data.loc[:, 'llgl']
gl = data.loc[:, 'gl']
# 截取指定的列
totaldata = data.loc[:, ['fs', 'llgl', 'gl']]
print("总长度：" + str(len(totaldata)))
fullData = totaldata[(totaldata['gl'] > 0)].copy()
print("不缺失长度：" + str(len(fullData)))
inputationData = totaldata[(totaldata['gl'] <= 0)].copy()
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
fullData = std.inverse_transform(fullData)
print(centers)
labels = cluster.labels_  # 每一行的标签

##还原成带lable数据 这里用不到
# list=fullData.tolist()
# for i in range(0,len(list)):
#     list[i].append(labels[i])
# withlabeldata=np.array(list)
# groupdata=pd.DataFrame(withlabeldata).groupby(3)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(fullData, labels)

# for i in range(0, len(fullData)):
#     #print(np.insert(fullData[i], 3, labels[i], axis=0))
#     list=np.array(fullData[i])
#     list.append(labels[i])

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

point = []
print('\r\n')
# 对缺失的数据填充
for i in range(0, len(inputationData)):  # i是索引 x是值
    item = inputationData.values[i]
    fs = item[0]
    llgl = item[1]
    gl = item[2]
    x = np.array([fs, llgl])
    print('要填补的是' + str(fs) + " " + str(llgl) + " " + str(gl))
    # 根据 风速和理论功率找最近的簇心，
    distancelist = []
    for j in range(0, len(centers)):
        center_fs = centers[j][0]
        center_llgl = centers[j][1]
        y = np.array([center_fs, center_llgl])
        # 风速理论功率 求距离
        xy = np.vstack([x, y])
        distance = pdist(xy, metric='euclidean')[0]
        distancelist.append(distance)
        # print("和本次簇心的距离是：" + str(distance))

    mindistance = np.min(np.asarray(distancelist))
    indexTuple = np.where(distancelist == mindistance)[0][0]
    print("簇心索引是" + str(indexTuple))
    # print("距离索引为" + str(indexTuple) + "的簇心最近，距离是" + str(mindistance))
    distancelist = []  # 重新置空

    # 再根据最近的簇心通过knn找k个邻居，求std和mean
    currentcenter = centers[indexTuple]
    dis, indexs = knn.kneighbors([currentcenter], 10)
    for index, va in enumerate(indexs[0]):
        point.append(fullData[va])  # 需不需要改成带理论功率的withllgldata，做到em再思考
    # 找出label等于indexTuple,也就是找出此簇心内的所有点
    # dataaroundcenter = dic[indexTuple]

    # print("有此次有"+str(len(dataaroundcenter))+"个点")
    mean = np.array(point)[:, 2].mean()
    std = np.array(point)[:, 2].std()
    point.append([fs, llgl, np.NaN])
    data_missing = pd.DataFrame(point)
    em_imputed = impy.em(data_missing.values, mean, std)
    emRes = em_imputed[len(em_imputed) - 1]
    if (emRes[2] < 0 or emRes[2] > currentcenter[2]):
        emRes[2] = currentcenter[2]
    print("预测的结果是：" + str(emRes))
    print("----------该次结束------------")
    print("\r\n")
    point = []
    totaldata.iloc[inputationData.index[i]] = [emRes[0], emRes[1] * 1000, emRes[2] * 1000]
    b = 0
# 再根据邻居算std和mean作为em的参数

# for j ,x in range(0, len(item)):
#     print(j)
a = np.column_stack(
    [date, totaldata.loc[:, 'fs'], totaldata.loc[:, 'llgl'], totaldata.loc[:, 'gl']])
pd.DataFrame(a).to_csv("data(no need commit)\\填充后.csv", index=False, header=['date', 'fs', 'llgl', 'gl'])  # 写入文件
