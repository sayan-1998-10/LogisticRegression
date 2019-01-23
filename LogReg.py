import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
epsilon = 1e-5

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dot_product(X,theta):
    return np.dot(X,theta)

def cost(X,theta,Y):
    hx = dot_product(X,theta)
    J = (1/len(X)) * ((-Y.T @ np.log(sigmoid(hx)+epsilon)) -(1-Y).T@ np.log(1-sigmoid(hx)+epsilon))
    return J
def grad_descent_runner(iterations,X,Y,theta):

    for i in range(iterations):
         [grad,theta] = grad_descent(X,Y,theta)

    return [grad,theta]
def grad_descent(X,Y,theta):
    alpha = 0.001
    hx = dot_product(X,theta)
    grad = (1/len(X))*(X.T @ (sigmoid(hx)-Y))

    theta = theta - alpha*grad
    return [grad,theta]

def check(theta,X):
    p = np.zeros((len(X),1))
    h=(sigmoid(np.dot(X,theta)))
    for i in range(len(X)):
        if (h[i,0]>=0.5):
            p[i,0] = 1
        else:
            p[i,0] = 0
    return p
def plotting(X,Y):
    plt.scatter(X['1'],X['2'],c=Y['0'],marker='^',alpha=0.6)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
    plt.show()
def main():

    #loading data
    marks = pd.read_csv("C:/Users/europ/Desktop/machine-learning-ex2/ex2/ex2data1.txt",header=None)
    marks['3']= 1 #adding bias term

    #separating X and Y
    Y = marks.iloc[:,-2:-1]
    X = marks.iloc[:,:-2]
    df = marks.iloc[:,-1:]
    X = pd.concat((X,df),axis=1)
    X = X[['3',0,1]]
    X.columns=['0','1','2']
    Y.columns=['0']

    #defining weights
    theta = np.array([[-24],[0.2],[0.2]])
    #theta = np.zeros((3,1))

    iterations = 1000;
    print("Cost at starting is {0}".format(cost(X,theta,Y)))
    [grad,theta]=grad_descent_runner(iterations,X,Y,theta)
    print(theta)
    print(grad)
    print("Cost now after {0} iterations is {1}".format(iterations,cost(X,theta,Y)))

    check(theta,X)
    plotting(X,Y)

if __name__== '__main__' :
    main()

