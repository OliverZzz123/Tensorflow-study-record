import numpy as np
import pandas as pd
import pylab

def compute_error(b,w,data):  #计算误差
    totalError = 0
    x = data['r1']
    y = data['r2']
    totalError = (y-w*x-b)**2
    totalError = np.sum(totalError,axis=0)
    return totalError/float(len(data))

def optimizer(data,starting_b,starting_w,learning_rate,num_iter):  #优化器：梯度下降
    b = starting_b
    w = starting_w
    #梯度下降
    for i in range(num_iter):
        #通过执行以下操作，用新的更精确的b和m更新b和m
        # thie gradient step
        b,w =compute_gradient(b,w,data,learning_rate)
        if i%100==0:
            print('iter {0}:error={1}'.format(i,compute_error(b,w,data)))
    return [b,w]

def compute_gradient(b_current,w_current,data ,learning_rate): #计算梯度做参数更新

    b_gradient = 0
    w_gradient = 0

    N = float(len(data))
    x = data['r1']
    y = data['r2']
    b_gradient = -(2/N)*(y-w_current*x-b_current)
    b_gradient = np.sum(b_gradient,axis=0)
    w_gradient = -(2/N)*x*(y-w_current*x-b_current)
    w_gradient = np.sum(w_gradient,axis=0)
    #使用偏导数更新b和m值

    new_b = b_current - (learning_rate * b_gradient)
    new_w = w_current - (learning_rate * w_gradient)
    return [new_b,new_w]


def plot_data(data,b,w):
    x = data['r1']
    y = data['r2']
    y_predict = w*x+b
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()


def Linear_regression():
    # get train data
    data = pd.read_excel('data.xlsx')

    #定义超参数
    #学习率用于更新梯度
    #定义要迭代的数字
    #定义  y =wx+b
    learning_rate = 0.001
    initial_b =0.0
    initial_w = 0.0
    num_iter = 1000

    #训练模型
    #print b w error
    print('初始参数:\n initial_b = {0}\n intial_w = {1}\n error of begin = {2} \n'.format(initial_b,initial_w,compute_error(initial_b,initial_w,data)))

    #optimizing b and m
    [b ,w] = optimizer(data,initial_b,initial_w,learning_rate,num_iter)

    #print final b m error
    print('\n最终参数:\n b = {1}\n w={2}\n error of end = {3} \n'.format(num_iter,b,w,compute_error(b,w,data)))

    #plot result
    plot_data(data,b,w)

if __name__ =='__main__':
    Linear_regression()