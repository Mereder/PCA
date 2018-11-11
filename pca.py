from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    # 此处将读取数据全都转换为 float  NAN 从str变成 float 的nan
    datArr = [list(map(float,line)) for line in stringArr]
    fr.close()
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

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # isnan 判断是否是 nan  矩阵.A 将矩阵转为 array数组
        # nonzero(array) 判断np.array数组中非0元素的情况  返回的第一个数组 表示非0元素的行 第二个数组表示非0元素所在的列
        # 这里取[0] 即 非0元素的行
        # 本句话将去除NAN元素 然后将其他元素根据下边求mean  求第i列的mean
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        # 将NAN的地方的值 用mean 代替 数据处理方式
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


if __name__ == '__main__':
    dataMat = replaceNanWithMean()

    lowData, reconDat = pca(dataMat,2)

    print(lowData)
    # print(reconDat)
