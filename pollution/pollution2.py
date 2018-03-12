# author cj  time:2017/12/18
# 数据预处理,归一化，数据前六个为特征1.每小时车数 2.地面温度(摄氏度）。 3.风速(米/秒)。4.地面温度的差值(摄氏度)，
#  5.风向(0 - 360度)。6.每天的时间（小时）,7.第几天。 最后一个为结果：PM10（粒子）。
# pollution 6个，pollution1 5个 ,polltion2 7个
from sklearn.model_selection import KFold
import numpy as np
from pollution.gradient import *
data=np.loadtxt(fname='pollution2.txt')
tem=data.copy()
X_data=tem[:,0:7]
X_label=tem[:,7]
print(tem)

skf=KFold(n_splits=4,shuffle=True)#4折交叉验证
for train_index,test_index in skf.split(X_data,X_label):
    X_trainData, X_trainLabel = X_data[train_index], X_label[train_index]#训练集的数据，训练集的结果
    X_testData, X_testLabel = X_data[test_index], X_label[test_index]#测试集的数据，测试集的结果

    m_train = np.size(X_trainLabel)  # 训练集的数量
    m_test = np.size(X_testLabel)  # 测试集的数量
    m_feature = np.size(X_trainData, axis=1)
    print('trainnumber:', m_train)
    print('testnumber:', m_test)
    X_trainData = np.hstack((np.ones(shape=(m_train, 1)), X_trainData)) #给训练数集X0设置为1，方便矩阵运算
    X_testData = np.hstack((np.ones(shape=(m_test, 1)), X_testData))# 给测试数集X0设置为1，方便矩阵运算

    theta = np.zeros(shape=(m_feature + 1, 1)).flatten()  # 初始化权值，为6行一列
    J = computeCost(X_trainData, X_trainLabel, theta)
    print('Testing the cost function=', J)

    print('\nRunning normalEqn ...\n')
    theta= normalEqn(X_trainData, X_trainLabel, theta)
    print('renew theta is:', theta)

    # print('see the cost is decreasing:',J_history)
    print('the average cost is:', averageCost(X_testData, X_testLabel, theta))
    predict = np.dot(X_testData, theta)
    print(predict[0:10], X_testLabel[0:10])
    plotData(predict[0:50], X_testLabel[0:50])