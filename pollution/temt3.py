# author cj  time:2018/1/4
# author cj  time:2017/12/27
# 核岭回归（KRR）结合岭回归（线性最小二乘与l2范数正则化）与核技巧。
# 因此它学习了由各个内核和数据引起的空间中的线性函数。
# pollution 6个，pollution1 5个 ,polltion2 7个特征，1个结果

from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np

from pollution.gradient import *
data=np.loadtxt(fname='pollution2.txt')
mu=np.average(data[:,0:7],axis=0)
sigma=np.std(data[:,0:7],axis=0)
data[:,0:7]=Z_ScoreNormalization(data[:,0:7],mu,sigma)
tem=data.copy()
X_data=tem[:,0:7]
X_label=tem[:,7]
print(tem)

skf=KFold(n_splits=5)#5折交叉验证
for train_index,test_index in skf.split(X_data,X_label):
    X_trainData, X_trainLabel = X_data[train_index], X_label[train_index]#训练集的数据，训练集的结果
    X_testData, X_testLabel = X_data[test_index], X_label[test_index]#测试集的数据，测试集的结果
    # X_testLabel=np.reshape(X_testLabel,newshape=(np.size(X_testLabel),1))


    reg = linear_model.Ridge()

    reg.fit(X_trainData,X_trainLabel)
    pre=reg.score(X_testData,X_testLabel)
    m_test = np.size(X_testLabel)  # 测试集的数量
    theta1=reg.coef_
    theta2=reg.intercept_


    theta=np.hstack((theta2,theta1))
    X_testData = np.hstack((np.ones(shape=(m_test, 1)), X_testData))

    print('the average cost is:', averageCost(X_testData, X_testLabel, theta))
    predict = np.dot(X_testData, theta)
    print(predict[0:10])
    print(X_testLabel[0:10])

    plotData(predict[0:50], X_testLabel[0:50])



