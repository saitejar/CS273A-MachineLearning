# To force divisions always be considered as float divisions
from __future__ import division 
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_digits
import itertools
from memory_profiler import profile
from operator import pos
import matplotlib
from matplotlib import pyplot as plt
from prettytable import PrettyTable

np.random.seed(776)



def tanh(inp, derivative=False):
    signal = np.tanh( inp )
    if derivative:
        return 1-np.power(signal,2)
    else:
        return signal
def sigmoid(inp, derivative=False):
    signal = expit( inp )
    if derivative:
        return np.multiply(signal, 1-signal)
    else:
        return signal

def identity(inp, derivative = False):
    if derivative:
        return 1
    else:
        return inp
    
def scale(X, eps = 0.001):
    #scale the data points 
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

class data:
    pass


class neuralNet:
    def __init__(self, X, y, test, y_test, numInputs, numOutputs, numHiddenLayers=1, numHiddenNodesInEachLayer=[30], transferFunc=0): 
        self.X = X 
        self.y = y
        self.test = test
        self.y_test = y_test
        #assert len(X[0])==numInputs, "give proper data, numofInputs not matching with the X"
        #assert len(y[0])==numOfOutputs, "give proper data, numOfOutputs not matching with y"
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.numHiddenLayers = numHiddenLayers
        self.numHiddenNodesInEachLayer = numHiddenNodesInEachLayer
        self.numHiddenNodesInEachLayer.append(numOutputs)
        self.numOfDataPoints = len(X)

        if transferFunc==0:
            self.transferFunc = [ tanh ]*self.numHiddenLayers+ [sigmoid]
        else:
            self.transferFunc = transferFunc
        
        
        if self.numHiddenLayers == 0:
            self.numOfWeights = (self.numInputs+1)*self.numOutputs
        else:
            self.numOfWeights = (self.numInputs+1)*self.numHiddenNodesInEachLayer[0]
            self.numOfWeights += (self.numHiddenNodesInEachLayer[self.numHiddenLayers-1]+1)*self.numOutputs
            for hdLayer in range(self.numHiddenLayers-1):
                self.numOfWeights += (self.numHiddenNodesInEachLayer[hdLayer]+1)*(self.numHiddenNodesInEachLayer[hdLayer+1])
        
        self.initWeights()
        self.layeredWeights = self.layerize()
        self.biasPos = []
        offset = 0
        self.biasPos = self.biasPos + range(offset,offset+self.numHiddenNodesInEachLayer[0])
        offset+=(self.numInputs+1)*self.numHiddenNodesInEachLayer[0]
        for i in xrange(1,self.numHiddenLayers+1):
            print i
            self.biasPos = self.biasPos + range(offset,offset+self.numHiddenNodesInEachLayer[i],1)
            offset+=(self.numHiddenNodesInEachLayer[i-1]+1)*(self.numHiddenNodesInEachLayer[i])
        print self.biasPos

    def initWeights(self):
        self.weights = np.random.normal(0.001,0.1,self.numOfWeights)

    def initDerivatives(self):
        self.derivatives = np.zeros(self.numOfWeights)
    
    def layerize(self, w = True):
        if w:
            item = self.weights
        else:
            item = self.derivatives
        layered = []
        curIndex = 0
        if self.numHiddenLayers == 0:
            layered.append(np.reshape(item, (self.numInputs+1,self.numOutputs)))
        else:
            layer0 = np.reshape(item[0:((self.numInputs+1)*(self.numHiddenNodesInEachLayer[0]))], ((self.numInputs+1),(self.numHiddenNodesInEachLayer[0])))
            layered.append(layer0)
            curIndex += ((self.numInputs+1)*(self.numHiddenNodesInEachLayer[0]))
            for hdLayer in xrange(self.numHiddenLayers-1):
                layered.append(np.reshape(item[curIndex:curIndex+(self.numHiddenNodesInEachLayer[hdLayer]+1)*(self.numHiddenNodesInEachLayer[hdLayer+1])],(self.numHiddenNodesInEachLayer[hdLayer]+1,self.numHiddenNodesInEachLayer[hdLayer+1])))
                curIndex += (self.numHiddenNodesInEachLayer[hdLayer]+1)*(self.numHiddenNodesInEachLayer[hdLayer+1])
            layerN = np.reshape(item[curIndex:],(self.numHiddenNodesInEachLayer[self.numHiddenLayers-1]+1,self.numOutputs))
            curIndex += (self.numHiddenNodesInEachLayer[self.numHiddenLayers-1]+1) * (self.numOutputs)
            layered.append(layerN)
        return layered
    
    def feedForward(self, dataPoints, numOfPoints):
        dataPoints = dataPoints.T
        #print dataPoints.shape
        #print (np.ones((1,self.miniBatchSize))).shape
        dataPoints = np.concatenate((np.ones((1,numOfPoints)), dataPoints), axis = 0)
        self.activationsAtLayers = []
        self.activationsAtLayers.append(dataPoints)
        for layer in xrange(self.numHiddenLayers+1):
            transferFunc = np.vectorize(self.transferFunc[layer])
            #print self.layeredWeights[layer].shape
            #print self.activationsAtLayers[layer].shape
            
            activations = transferFunc(np.dot(self.layeredWeights[layer].T, self.activationsAtLayers[layer]))
            #print np.dot(self.layeredWeights[layer].T, self.activationsAtLayers[layer])
            
            if layer != self.numHiddenLayers:
                activations = np.concatenate((np.ones((1,numOfPoints)),activations), axis = 0)
            self.activationsAtLayers.append(activations)
    
    def backPropagationAlgorithm(self, dataPoints, T):
        T = T.T
        self.feedForward(dataPoints,self.miniBatchSize)
        transferFuncPrime = np.vectorize(self.transferFunc[self.numHiddenLayers])
        activationPrimes = transferFuncPrime(self.activationsAtLayers[self.numHiddenLayers+1], derivative = True)
        self.delta = []
        self.delta.append((self.activationsAtLayers[self.numHiddenLayers+1]-T) * activationPrimes)
        self.derivatives = []
        for layer in xrange(self.numHiddenLayers,0,-1):
            transferFuncPrime = np.vectorize(self.transferFunc[layer-1])
            self.derivatives.insert(0, np.einsum('ik,jk->ij',self.activationsAtLayers[layer],self.delta[0]))
            activationPrimes = transferFuncPrime(self.activationsAtLayers[layer][1:,:], derivative = True)
            deltaThisLayer = np.dot(self.layeredWeights[layer][1:,:],self.delta[0]) * activationPrimes
            self.delta.insert(0, deltaThisLayer)
        
        self.derivatives.insert(0, np.einsum('ik,jk->ij',self.activationsAtLayers[0],self.delta[0]))
        self.derivatives = np.concatenate([i.flatten() for i in self.derivatives])
        
        self.weights -= self.alpha * self.derivatives * (1/self.miniBatchSize)
        mask = np.ones(self.numOfWeights, dtype=bool)
        mask[self.biasPos] = False
        self.derivatives[mask] += self.Lambda * self.weights[mask] * self.alpha
        
        #print self.weights[10]
        self.layeredWeights = self.layerize()
        
        #print self.activationsAtLayers[2]
        
        #self.feedForward(self.test, 1000)
        #print 'pred = ', np.argmax(self.activationsAtLayers[self.numHiddenLayers+1], 0) 
        #jprint 'act = ', self.y_test
        #y_pred = 1 * (self.activationsAtLayers[self.numHiddenLayers+1] < 0.5)
        #print self.activationsAtLayers[self.numHiddenLayers+1]
        #print 'class error = ',np.sum(y_pred == self.y_test)/20
        
     
    def train(self, miniBatchSize = 50, epochs = 10000, alpha = 0.01, Lambda = 0.0001):
        self.Lambda = Lambda
        self.alpha = alpha
        self.miniBatchSize = miniBatchSize
        pos = 0
        for i in range(epochs):
            #print i, " error = ",
            a = pos%(self.numOfDataPoints)
            b = (pos+miniBatchSize)%self.numOfDataPoints
            if b==0:
                b=self.numOfDataPoints
            if b>a:
                self.backPropagationAlgorithm(self.X[a:b], self.y[a:b])
                self.feedForward(self.X[a:b], miniBatchSize) 
                #print self.activationsAtLayers[self.numHiddenLayers+1][:,0][0] - self.y[a:b].T[:,0][0]
                #print np.mean(np.sqrt(np.mean((self.activationsAtLayers[self.numHiddenLayers+1] - self.y[a:b].T)**2, 0))/miniBatchSize)
                #print np.mean((np.mean(np.abs(self.activationsAtLayers[self.numHiddenLayers+1] - self.y[a:b].T), 0)))
            else:
                self.backPropagationAlgorithm(np.vstack((self.X[a:,],self.X[0:b,])),np.vstack((self.y[a:,],self.y[0:b,])))
                self.feedForward(np.vstack((self.X[a:],self.X[0:b])), miniBatchSize)
                #print np.mean(np.mean(np.abs(self.activationsAtLayers[self.numHiddenLayers+1] - np.vstack((self.y[a:],self.y[0:b])).T), 0))
            pos+=self.miniBatchSize
        self.feedForward(self.test, self.numOfDataPoints)
        mean = np.mean((np.mean(np.abs(self.activationsAtLayers[self.numHiddenLayers+1] - self.y.T), 0)))
        v = np.std((np.mean(np.abs(self.activationsAtLayers[self.numHiddenLayers+1] - self.y.T), 0)))
        
        return [mean,v]
        #y_predict = 1 * (self.activationsAtLayers[self.numHiddenLayers+1] > 0.5)
        self.initWeights()
        self.layeredWeights = self.layerize()
#         print type(y_predict)
#         print y_predict.shape
#         print self.y.T.shape
#         print y_predict
#         print self.y.T
        
        #return np.sum(y_predict==self.y.T)/self.numOfDataPoints
        
        
def generateOneOn(n_data=100, n_dim=9):
    p_threshold = 2 / n_dim
    p_matrix = np . random . rand ( n_data , n_dim )
    x_matrix = 1 * (p_matrix < p_threshold)
    y = (1 * ( np .sum ( x_matrix , 1) == 1))
    y = y.reshape((n_data,1))
    
    p_matrix_test = np . random . rand ( n_data , n_dim )
    x_matrix_test = 1 * (p_matrix_test < p_threshold)
    y_test = (1 * ( np .sum ( x_matrix_test , 1) == 1))
    y_test = y_test.reshape((n_data,1))
    return [x_matrix, y, x_matrix_test, y_test]

def generateAutoEncoderData(n_data=1000, n_dim_full = 100, n_dim_limited = 30):
    eigenvals_big = np . random . randn ( n_dim_limited ) + 3
    eigenvals_small = np .abs( np . random . randn ( n_dim_full - n_dim_limited )) * .1
    eigenvals = np . concatenate ([ eigenvals_big , eigenvals_small ])
    
    diag = np . diag ( eigenvals )
    q , r = np . linalg . qr ( np . random . randn ( n_dim_full , n_dim_full ))
    cov_mat = q . dot ( diag ). dot ( q . T )
    mu = np . zeros ( n_dim_full )
    x = np . random . multivariate_normal ( mu , cov_mat , n_data )
    z = np . random . multivariate_normal ( mu , cov_mat , n_data )
    return [x, x, z]

if __name__ == "__main__":
    [x_matrix, y,x_test, y_test] = generateOneOn(100,9)
    #print y
    #[x_matrix, y,z] = generateAutoEncoderData()
    #np.savetxt('in.csv', x_matrix)
    #np.savetxt('out.csv',y)
    print x_matrix.shape[1]
    print y.shape[1]
    dimInput = x_matrix.shape[1]
    dimOutput = y.shape[1]
    numOfDataPoints = x_matrix.shape[0]
    table = PrettyTable(["iterations\ Lambda",'0.1','0.01','0.001'])
    rows = []
    NN = neuralNet(x_matrix,y,x_test,y_test,dimInput,dimOutput,1,[15],[tanh]*1+ [identity])
    alphas = [0.5,0.1,0.01,0.001]
    epochs = [500,1000,2000,5000,10000]
    decay = [0.01, 0.001, 0.0001]
    alpha = 0.01
    for epoch in epochs:
        print epoch
        rows.append(epoch)
        for Lambda in decay:
            print Lambda
            [mu,v] = NN.train(30,epoch,alpha,Lambda)
            print mu,v
            s = round(mu,4),round(v,4)
            rows.append(s)
        table.add_row(rows)
        rows=[]
    print table
    
    
