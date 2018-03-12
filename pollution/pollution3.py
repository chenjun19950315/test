# author cj  time:2018/1/2

# author cj  time:2018/1/2

import numpy as np
from sklearn.model_selection import KFold
from pollution.gradient import *
from sklearn import linear_model

data=np.loadtxt(fname='pollution2.txt')
mu=np.average(data[:,0:7],axis=0)
sigma=np.std(data[:,0:7],axis=0)
data[:,0:7]=Z_ScoreNormalization(data[:,0:7],mu,sigma)
tem=data.copy()
X_data=tem[:,0:7]
X_label=tem[:,7]
print(tem)

skf=KFold(n_splits=4,shuffle=True)#4折交叉验证
for train_index,test_index in skf.split(X_data,X_label):
    X_trainData, X_trainLabel = X_data[train_index], X_label[train_index]#训练集的数据，训练集的结果
    X_testData, X_testLabel = X_data[test_index], X_label[test_index]#测试集的数据，测试集的结果
    # X_testLabel=np.reshape(X_testLabel,newshape=(np.size(X_testLabel),1))


    reg=linear_model.Lasso(alpha=0.03,max_iter=10000,tol=0.00001)
    reg.fit(X_trainData,X_trainLabel)
    m_test = np.size(X_testLabel)  # 测试集的数量
    theta1=reg.coef_
    theta2=reg.intercept_


    theta=np.hstack((theta2,theta1))

    X_testData = np.hstack((np.ones(shape=(m_test, 1)), X_testData))
    print('测试集数量：',m_test)
    print('the average cost is:', averageCost(X_testData, X_testLabel, theta))
    predict = np.dot(X_testData, theta)
    print(predict[0:10], X_testLabel[0:10])
    plotData(predict[0:30], X_testLabel[0:30])
