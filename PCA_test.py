from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def loadData():
    dataList = []
    with open('imports-85.data','r') as fr:
        for line in fr.readlines():
            newline = line.strip().split(',')[9:]
            del (newline[5])
            del (newline[5])
            del (newline[6])
            if '?' in newline:
                continue
            else:
                dataList.append(newline)
    datArr = [list(map(float,line)) for line in dataList]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)  # axis = 0 对各列求均值 返回1*n 矩阵
    meanRemoved = dataMat - meanVals # 将数据中心化
    # 其中rowvar是布尔类型。默认为true是将列作为独立的变量、如果是flase的话，则将行作为独立的变量
    covMat = cov(meanRemoved, rowvar=0) # 求X.T*X 即 协方差
    eigVals,eigVects = linalg.eig(mat(covMat))  # 求特征向量和特征值
    print(eigVals)
    eigValInd = argsort(eigVals)            # 将x中的元素从小到大排列，提取其对应的index(索引)，sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # topNfeat = 2 则取最大的两个，最后一个-1 代表逆序每隔开一个  cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects    #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

if __name__ == '__main__':
    dataMat = loadData()
    lowDMat, reconMat = pca(dataMat, 1)
    lowData, reconDat = pca(dataMat, 2)

    print(lowData)
    # print(reconDat)

def plotData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def plottingPercent(eigVals):
    print(eigVals)
    # 换算下百分比
    precent = (eigVals / sum(eigVals))* 100
    y = np.array(precent)
    x = np.array(range(14))

    plt.plot(x,y,'r',lw=1)# 4 line w
    plt.show()
