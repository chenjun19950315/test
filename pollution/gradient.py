# author cj  time:2017/12/17

#Z_Score归一化函数
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# 归一化函数
def Z_ScoreNormalization(data,mu,sigma):
    data = (data - mu) / sigma
    return data

def computeCost(X, y, theta):
    m = np.size(y,axis=0) #训练集数量
    J = 0
    hypothesis = np.dot(X ,theta)
    J = np.sum((hypothesis-y)**2)/(2*m)
    return J
# 梯度下降算法
def gradientDescent(X, y, theta, alpha, num_iters):
    m = np.size(y, axis=0)#训练集数量
    J_history = np.zeros(shape=(num_iters,1)).flatten() #记录平方误差
    n = np.size(X, 1)
    for iter1 in range(0,num_iters):
        H=np.dot(X,theta)
        t= np.zeros(shape=(n,1)).flatten()

        for i in range(0,m):
            t1 =X[i,:].T
            t=t+(H[i]-y[i])*t1;

        J_history[iter1] = computeCost(X, y, theta)
        theta = theta - (alpha * t) / m
    return theta,J_history
# 画图函数
def plotData(y1,y2):
    plt.plot(y1, color="red",label='forecast')
    plt.ylabel('airplane sound Y')


    plt.plot(y2, color="black",label='fact')
    plt.legend()
    plt.title("Comparison of predicted and actual values")
    plt.show()
#     平均误差
def averageCost(X, y, theta):
    m = np.size(y) #训练集数量

    hypothesis = np.dot(X ,theta)

    J = np.sum(abs((hypothesis-y)))/m
    return J


def averageCost1(X, y, theta):
    m = np.size(y)  # 训练集数量

    hypothesis = np.dot(X, theta)
    y = np.reshape(y, (m, 1))

    print(np.hstack((hypothesis, y)))
    J = np.sum(abs((hypothesis - y))) / m
    return J
# 正规方程
def normalEqn(X, y, theta):
   theta =np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))),X.T),y)
   return theta