#!/usr/bin/python

import numpy as np
import math
from prettytable import PrettyTable

class DataGen:
    clusterCenterR = 0
    clusterCenterB = 0
    pickCluster = False
    alpha = 0
    def __init__(self, numOfPointsTrain = 1000, numOfPointsTest = 1000, dimOfVec = 9, alpha = 0.8, flag = 0, mu1=0, mu2=0):
        self.numOfPointsTrain = numOfPointsTrain
        self.numOfPointsTest = numOfPointsTest
        self.dimOfVec = dimOfVec
        if flag==0:
            self.centers = self.pickClusterCenters()
        else:
            self.centers = (mu1, mu2)
        self.alpha = alpha #np.random.random()
        self.train = self.generateTrainData()
        self.test = self.generateTestData()
    
        
    def __del__(self):
        pass
        
    def pickClusterCenters(self):
        [clusterCenterR, clusterCenterB] = np.random.multivariate_normal(np.zeros(self.dimOfVec), np.identity(self.dimOfVec), 2)
        return (clusterCenterR, clusterCenterB)
    
    def generateTrainData(self):
        pointsR = np.random.multivariate_normal(self.centers[0], self.alpha * np.identity(self.dimOfVec), self.numOfPointsTrain/2)
        pointsB = np.random.multivariate_normal(self.centers[1], self.alpha * np.identity(self.dimOfVec), self.numOfPointsTrain/2)
        X = np.concatenate((pointsR, pointsB), 0)
        #print (X.shape)
        #print X
        Y = np.array([-1]*(self.numOfPointsTrain/2)+ [1]*(self.numOfPointsTrain/2))
        
        return (X,Y)
    
    def generateTestData(self):
        
        pointsR = np.random.multivariate_normal(self.centers[0], self.alpha * np.identity(self.dimOfVec), self.numOfPointsTest/2)
        pointsB = np.random.multivariate_normal(self.centers[1], self.alpha * np.identity(self.dimOfVec), self.numOfPointsTest/2)
        X = np.concatenate((pointsR, pointsB), 0)
        y = np.array([-1]*(self.numOfPointsTest/2) + [1]*(self.numOfPointsTest/2))
            
        return (X,y)
    
def zeroOneLoss(y, y_est):
    return float(sum([1 for i,j in zip(y, y_est) if i==j]))/len(y) * 100
    
def fisherLDATrain(X,y):
    #numOfDim = X.shape[1]
    Nr = sum(1 for i in xrange(len(y)) if y[i]==-1)
    Nb = sum(1 for i in xrange(len(y)) if y[i]==1)
    N = Nr + Nb
    
    muR = sum(X[i] for i in xrange(N) if y[i]==-1)/Nb
    muB = sum(X[i] for i in xrange(N) if y[i]==1)/Nr

    Xr = X[y == -1, :]
    Sw = np.transpose(np.mat(Xr-muR)) * np.mat(Xr-muR)
    
    Xb = X[y == 1, :]
    Sw += np.transpose(np.mat(Xb-muB)) * np.mat(Xb-muB)

    w = np.linalg.inv(Sw) * np.transpose(np.mat(muB-muR))
    w = w/np.linalg.norm(w)
    projections = np.mat(X) * np.mat(w)
    P = np.array(projections).flatten()
    
    Pr = P[y==-1]
    Pr_mean = np.mean(Pr)
    Pr_sigma = float(Nr-1)/Nr * np.var(Pr)
    
    Pb = P[y==1]
    Pb_mean = np.mean(Pb)
    Pb_sigma = float(Nb-1)/Nb * np.var(Pb)
    return (w, Pr_mean,Pr_sigma, Pb_mean, Pb_sigma)
    
def mapping(val):
    if val==True:
        return -1
    else:
        return 1

def fisherLDATest(X, w, Pr_mean,Pr_sigma, Pb_mean, Pb_sigma):
    Xw = X * w
    Xw, = np.array(Xw.T)
    Xwr = Xw - Pr_mean
    Xwb = Xw - Pb_mean
    dr = 2*Pr_sigma*Pr_sigma
    db = 2*Pb_sigma*Pb_sigma
    lr = math.log(1/Pr_sigma)
    lb = math.log(1/Pb_sigma)
    Xwr = lr - (np.square(Xwr))/dr
    Xwb = lb - (np.square(Xwb))/db
    vmap = np.vectorize(mapping)
    y_est = vmap(Xwr>Xwb)
    return y_est

def perceptron(X,y,noOfIterations):
    #nOfDim = X.shape[1] # no of dimensions 
    N = X.shape[0]   #no of training examples
    learning_rate = 0.1
    w = np.random.random((X.shape[1]+1))
    bias = 1
    
    for i in xrange(noOfIterations*N):
        fx = np.inner(X[i%N],w[1:]) + w[0]*bias
        if fx >0:
            yi = 1
        else:
            yi = -1
        di = y[i%N]
        w = w + learning_rate * (di-yi) * (np.append([1], X[i%N]))
    
    return (w, bias)

def perceptronTest(X,w,bias):
    #nOfDim = X.shape[1] # no of dimensions 
    Xw = X * (np.mat(w[1:]).T)
    Xw, = np.array(Xw.T)
    
    Xw = Xw + w[0]*bias
    vmap = np.vectorize(mapping)
    y_est = vmap(Xw < np.zeros(X.shape[0]))

    return y_est
    
def main():
    open('output.txt','w')
    
# Code for 1B
    with open('output.txt','a') as f:
        f.write("1. Each entry of the tables below has two components, first one is mean accuracy, second is deviation of accuracy\n")
        f.write("2. First table corresponds to 1A, second to 1B, third 2B, fourth 2C\n")
        f.write("\n\n-------------TABLE FOR 1B - number of data points in training vs Alpha--------------\n")
    f.close()
    numOfDataSets = 50
    alpha_start_diff = 0.1

    header = ['N_train\Alpha']
    header.extend([str(round(x * alpha_start_diff, 2)) for x in range(1, 10)])
    
    table = PrettyTable(header)
    table.align["N_train\Alpha"] = "l" # Left align city names
    for N in xrange(50, 501, 50):
        rowTable = [str(N)]
        for alpha in [round(x * alpha_start_diff, 2) for x in range(1, 10)]:
            testAccuracy = [0]*numOfDataSets
            for dataSetNum in xrange(numOfDataSets):
                dataSet = DataGen(N,N,9,alpha)
                (w, Pr_mean, Pr_sigma, Pb_mean, Pb_sigma) = fisherLDATrain(dataSet.train[0], dataSet.train[1])
                y_est = fisherLDATest(dataSet.test[0], w, Pr_mean, Pr_sigma, Pb_mean, Pb_sigma)
                y_est = [int(i) for i in y_est]
                y = dataSet.test[1]
                s = 0
                for i in xrange(len(y)):
                    if y[i]==y_est[i]:
                        s = s+1           
                testAccuracy[dataSetNum] = (float(s)/len(y) * 100)
            mean = np.mean(testAccuracy)
            mean = round(mean, 2)
            stD = np.std(testAccuracy)
            stD = round(stD,2)
            
            #[mean, sigma] = [np.mean(testAccuracy), np.stD(testAccuracy)]
            rowTable.append(str(mean)+','+str(stD))
        table.add_row(rowTable)
    table_txt = table.get_string()
    with open('output.txt','a') as f:
        f.write(table_txt)
        f.write("\n")
    print table
    print '1b'


# Code for 1C
    numOfDataSets = 20
    header = ['N_test\N_Iterations']
    header.extend(range(1,11,1))
    with open('output.txt','a') as f:
        f.write("\n\n-------------TABLE FOR 1C - number of test data points vs number of iterations---------------\n")
    f.close()
    table = PrettyTable(header)
    table.align["N_test\N_Iterations"] = "l" # Left align city names
    for numOfDataPoints in xrange(50, 400, 50):
        rowTable = [numOfDataPoints]
        for numOfIteration in xrange(10): # no of iterations
                testAccuracy = [0]*numOfDataSets
                for dataSetNum in xrange(numOfDataSets):
                    dataSet = DataGen(numOfDataPoints, numOfDataPoints, 9, 0.1)
                    y = dataSet.train[1]
                    (w, bias) = perceptron(dataSet.train[0], y, numOfIteration)
                    
                    y_est = perceptronTest(dataSet.test[0], w, bias)
                    y = dataSet.test[1]
                    s = 0
                    for i in xrange(len(y)):
                        if y[i]==y_est[i]:
                            s = s+1
                    acc = float(s)/(len(y)) * 100
                    testAccuracy[dataSetNum] = acc 
    
                mean = round(np.mean(testAccuracy),2)
                stD = round(np.std(testAccuracy),2)
                rowTable.append(str(mean)+','+str(stD))
        table.add_row(rowTable)
    table_txt = table.get_string()
    print table 
    with open('output.txt','a') as f:
        f.write(table_txt)
        

# 2b
    mu1 = [1,0,0,1,1,1,0,0,1]
    mu2 = [1,1,1,1,0,0,1,0,0]
    alpha = 0.1
    numOfDataSets = 50
    alpha_start_diff = 0.1

    header = ['N_train\Alpha']
    header.extend([str(round(x * alpha_start_diff, 2)) for x in range(1, 10)])
    with open('output.txt','a') as f:
        f.write("\n\n-------------TABLE FOR 2B - number of data points in training vs Alpha--------------\n")
    f.close()
    table = PrettyTable(header)
    table.align["N_train\Alpha"] = "l" # Left align city names
    for N in xrange(50, 501, 50):
        rowTable = [str(N)]
        for alpha in [round(x * alpha_start_diff, 2) for x in range(1, 10)]:
            testAccuracy = [0]*numOfDataSets
            for dataSetNum in xrange(numOfDataSets):
                dataSet = DataGen(N,N,9,alpha,1,mu1,mu2)
                (w, Pr_mean, Pr_sigma, Pb_mean, Pb_sigma) = fisherLDATrain(dataSet.train[0], dataSet.train[1])
                y_est = fisherLDATest(dataSet.test[0], w, Pr_mean, Pr_sigma, Pb_mean, Pb_sigma)
                y_est = [int(i) for i in y_est]
                y = dataSet.test[1]
                s = 0
                for i in xrange(len(y)):
                    if y[i]==y_est[i]:
                        s = s+1           
                testAccuracy[dataSetNum] = (float(s)/len(y) * 100)
            mean = np.mean(testAccuracy)
            mean = round(mean, 2)
            stD = np.std(testAccuracy)
            stD = round(stD,2)
            #[mean, sigma] = [np.mean(testAccuracy), np.stD(testAccuracy)]
            rowTable.append(str(mean)+','+str(stD))
        table.add_row(rowTable)
    print table
    table_txt = table.get_string()
    with open('output.txt','a') as f:
        f.write(table_txt)
    print '2b'
    
# Code for 2C
    alpha = 0.1
    numOfDataSets = 20
    header = ['N_test\N_Iterations']
    header.extend(range(1,11,1))
    with open('output.txt','a') as f:
        f.write("\n\n-------------TABLE FOR 1C - number of test data points vs number of iterations---------------\n")
    f.close()
    table = PrettyTable(header)
    table.align["N_test\N_Iterations"] = "l" # Left align city names
    for numOfDataPoints in xrange(50, 400, 50):
        rowTable = [numOfDataPoints]
        for numOfIteration in xrange(10): # no of iterations
            testAccuracy = [0]*numOfDataSets
            for dataSetNum in xrange(numOfDataSets):
                dataSet = DataGen(numOfDataPoints,numOfDataPoints,9,alpha,1,mu1,mu2)
                y = dataSet.train[1]
                (w, bias) = perceptron(dataSet.train[0], y, numOfIteration)
                
                y_est = perceptronTest(dataSet.test[0], w, bias)
                y = dataSet.test[1]
                s = 0
                for i in xrange(len(y)):
                    if y[i]==y_est[i]:
                        s = s+1
                acc = float(s)/(len(y)) * 100
                testAccuracy[dataSetNum] = acc 

            mean = round(np.mean(testAccuracy),2)
            stD = round(np.std(testAccuracy),2)
            rowTable.append(str(mean)+','+str(stD))
        table.add_row(rowTable)
    table_txt = table.get_string()
    print table 
    with open('output.txt','a') as f:
        f.write(table_txt)
    pass

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
            
        
            
            
        
        
    
    
    
        
        
        
        
        