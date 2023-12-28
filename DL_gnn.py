from cmath import sqrt
from random import random, randrange
from matplotlib.pyplot import axis
import numpy as np
import math

from abc import ABC, abstractmethod
class Layer(ABC):
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []

  def setPrevIn(self,dataIn):
    self.__prevIn = dataIn
  
  def setPrevOut(self, out):
    self.__prevOut = out
  
  def getPrevIn(self):
    return self.__prevIn
  
  def getPrevOut(self):
    return self.__prevOut
  
  def backward(self, gradIn):
    sg = self.gradient()
    sgShape = np.shape(sg)
    
    if(len(sgShape) == 3):
        grad = np.zeros(( np.shape(gradIn)[0], np.shape(sg)[2]))
        for n in range( np.shape(gradIn)[0]):
            grad[n, :] = gradIn[n, :] @ sg[n, :, :]

        return grad

    return gradIn * sg

  @abstractmethod
  def forward(self,dataIn):
    pass
  
  @abstractmethod  
  def gradient(self):
    pass


class InputLayer(Layer):
  def __init__(self,dataIn):
    super().__init__()
    #Find mean and std-dev per feature.
    self.__meanX = np.mean(dataIn, axis=0)
    self.__stdX = np.std(dataIn, axis=0)
    
    self.__stdX[self.__stdX==0] = 1

  def forward(self,dataIn):
    self.setPrevIn(dataIn)

    #Z-Score the Data
    Z = (dataIn - self.__meanX) / self.__stdX
    self.setPrevOut(Z)
    return Z

  def gradient(self):
    pass

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()

        self.__W = np.random.uniform(size = (sizeIn, sizeOut), low = -0.1, high = 0.1)
        self.__B = np.random.uniform(size = (1, sizeOut), low = -0.1, high = 0.1)

        self.__p1 = 0.9
        self.__p2 = 0.999
        self.__s = np.zeros((sizeIn, sizeOut))
        self.__r = np.zeros((sizeIn, sizeOut))
        self.__epoch = 1

    def getWeights(self):
        return self.__W
    
    def setWeights(self, weights):
        self.__W = weights
    
    def getBias(self):
        return self.__B
    
    def setBias(self, bias):
        self.__B = bias
    
    def resetHyperParams(self):
        self.__p1 = 0.9
        self.__p2 = 0.999
        self.__s = np.zeros(np.shape(self.__W))
        self.__r = np.zeros(np.shape(self.__W))
        self.__epoch = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        #normal matrix multiplication uses np.dot
        Y = (dataIn @ self.__W) + self.__B
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        #should be a tensor
        gradient = np.transpose(self.__W)
        gradientShape = np.shape(gradient)

        numObservations = np.shape(self.getPrevIn())[0]
        
        tensor = np.zeros((numObservations, gradientShape[0], gradientShape[1]))
        for i in range(0, numObservations):
            tensor[i] = gradient

        return tensor
    
    def updateWeights(self, gradIn, eta = 0.0001):
        
        delta = pow(10, -8)
        N = np.shape(self.getPrevIn())[0]
        partialBias = np.sum(gradIn, axis = 0) / N
        partialWeights = (np.transpose(self.getPrevIn()) * gradIn) / N

        #ADAM 
        s = self.__p1*self.__s + (1-self.__p1)*partialWeights
        r = self.__p2*self.__r + (1-self.__p2)*( np.square(partialWeights))

        blendSR = (s/(1-np.power(self.__p1, self.__epoch))) / (np.sqrt( r / (1-np.power(self.__p2, self.__epoch))) + delta)

        self.__W -= eta*blendSR
        self.__B -= eta*partialBias
        # self.__W -= eta*partialWeights
        # self.__B -= eta*partialBias

        self.__epoch += 1

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = dataIn
        self.setPrevOut(Y)
        return Y
        
    def gradient(self):
        #gradient is equal to the identity matrix
        size = np.shape(self.getPrevIn())[1]
        return np.identity(size)

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient()) #hadamard product

class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)
        return Y
        
    def gradient(self):
        #gradient has zeros everywhere but on the diagonal
        #on the diagonals, it can be zero or one
        size = np.shape(self.getPrevIn())[1]
        gradientMatrix = np.identity(size)

        for index in range(0, size):
            previousMatrix = self.getPrevIn()

            if previousMatrix[0, index] < 0:
                gradientMatrix[index, index] = 0
        
        return gradientMatrix

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient()) #hadamard product

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = 1/(1+np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        #gradient has zeros everywhere but on the diagonal
        epsilon = pow(10, -7)
        size = np.shape(self.getPrevIn())[1]
        gradientMatrix = np.identity(size)

        for index in range(0, size):
            gzj = self.getPrevOut()[0, index]
            gradientMatrix[index, index] = gzj * (1-gzj) + epsilon
        
        return gradientMatrix

    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient()) #hadamard product

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        Y = np.exp(dataIn)
		# Sum going from left to right (sum of a row)
        denominator = np.sum(np.exp(dataIn), axis=1)

        for i, _ in enumerate(denominator):
            Y[i] = Y[i] / (denominator[i] + np.finfo(float).eps)
            
        self.setPrevOut(Y)
		
        return Y
        
    def gradient(self):
        #Should be a tensor
        preIn = self.getPrevIn()
        numObservations = np.shape(preIn)[0]
        size = np.shape(preIn)[1]
        
        tensor = np.zeros((numObservations, size, size))
        
        for o in range(0, numObservations):
            gradientMatrix = np.zeros((size,size))
            for i in range(0, size):
                gzi = self.getPrevOut()[0, i]
                for j in range(0, size):
                    gzj = self.getPrevOut()[0, j]

                    if i==j:
                        gradientMatrix[i, j] = gzj * (1-gzj)
                    else:
                        gradientMatrix[i, j] = -gzi * gzj
            
            tensor[o] = gradientMatrix
        return tensor


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.tanh(dataIn) #Using Tanh instead since it looks cleaner
        self.setPrevOut(Y)
        return Y
        
    def gradient(self):
        #gradient has zeros everywhere but on the diagonal
        epsilon = pow(10, -7)
        size = np.shape(self.getPrevIn())[1]
        gradientMatrix = np.identity(size)

        for index in range(0, size):
            gzj = self.getPrevOut()[0, index]
            gradientMatrix[index, index] = (1-(gzj*gzj)) + epsilon
        
        return gradientMatrix
    
    def backward(self, gradIn):
        return np.multiply(gradIn, self.gradient()) #hadamard product

class LeastSquares():
    def eval(self, y, yhat):
        return np.transpose(y-yhat) @ (y-yhat) / np.shape(y)[0]
    
    def gradient(self, y, yhat):
        return -2*(y-yhat)

class LogLoss():
    def eval(self, y, yhat):
        epsilon = pow(10, -7)
        return -(np.transpose(y)@np.log(yhat + epsilon) + np.transpose(1-y)@np.log(1-yhat + epsilon)) / np.shape(y)[0]
    
    def gradient(self, y, yhat):
        epsilon = pow(10, -7)
        return -((y-yhat) / (np.multiply(yhat,(1-yhat)) + epsilon))

class CrossEntropy():
    def eval(self, y, yhat):
        epsilon = pow(10, -7)
        yShape = np.shape(y)

        sum = 0
        for o in range(0, yShape[0]):
            sum += -(y[o]*np.log(np.transpose(yhat[o]) + epsilon) +
                        (1-y[o])*np.log(np.transpose(1 - yhat[o]) + epsilon))
        
        return sum / yShape[0]
    
    def gradient(self, y, yhat):
        epsilon = pow(10, -7)
        return -((y) / (yhat + epsilon))

class Network:
    def __init__(self):
        self.__layerList = []
        self.__objectiveFunction = LeastSquares()
        pass

    def addLayer(self, layer):
        self.__layerList.append(layer)
    
    def size(self):
        return len(self.__layerList)
    
    def getLayer(self, index):
        return self.__layerList[index]
    
    def setObjectiveFunction(self, function):
        self.__objectiveFunction = function

    def forward(self, inputData):
        index = 0
        layerOutput = inputData
        
        while (index < self.size()):
            layerOutput = self.__layerList[index].forward(layerOutput)
            index += 1
        
        return layerOutput
    
    def backwards(self, targetOutput, inputData, eta = 0.0001):
        grad = self.__objectiveFunction.gradient(targetOutput, inputData)
        for i in range(self.size()-1, 0, -1):
            newgrad = self.__layerList[i].backward(grad)

            if(isinstance(self.__layerList[i], FullyConnectedLayer)):
                self.__layerList[i].updateWeights(grad, eta)

            grad = newgrad

class GNN:

    def __init__(self):
        self.__layerList = []
        self.__fitnessValue = 0 #A measure of how well the network performed
        self.__mutationP = 0.02 #chance of mutation
        self.__mutationIntensity = 0.1 #How much to affect a weight or bias by if mutated

    def addLayer(self, layer):
        self.__layerList.append(layer)
    
    def size(self):
        return len(self.__layerList)
    
    def getLayer(self, index):
        return self.__layerList[index]

    def setFitnessValue(self, value):
        self.__fitnessValue = value

    def getFitnessValue(self):
        return self.__fitnessValue

    def setMutationProbability(self, value):
        self.__mutationP = value
    
    def getMutationProbability(self):
        return self.__mutationP

    def getMutationIntensity(self):
        return self.__mutationIntensity
    
    def setMutationIntensity(self, value):
        self.__mutationIntensity = value

    def forward(self, inputData):
        index = 0
        layerOutput = inputData
        
        while (index < self.size()):
            layerOutput = self.__layerList[index].forward(layerOutput)
            index += 1
        
        return layerOutput
    
    def crossover(self, other):
        newNetwork = GNN()
        newNetwork.setMutationIntensity( self.getMutationIntensity() )
        newNetwork.setMutationProbability( self.getMutationProbability() )

        for i in range(0, self.size()):
            v1 = self.getLayer(i)
            if(isinstance(v1, FullyConnectedLayer)):
                v2 = other.getLayer(i)

                w1 = v1.getWeights()
                w2 = v2.getWeights()

                b1 = v1.getBias()
                b2 = v2.getBias()

                wShape = np.shape(w1)
                bShape = np.shape(b1)

                newLayer = FullyConnectedLayer(wShape[0], wShape[1])

                wSplitPoint = randrange(0, wShape[1])
                bSplitPoint = randrange(0, bShape[1])

                finalWMatrix = np.append( np.hsplit(w1, [wSplitPoint])[0], np.hsplit(w2, [wSplitPoint])[1], axis=1 )
                finalBMatrix = np.append( np.hsplit(b1, [bSplitPoint])[0], np.hsplit(b2, [bSplitPoint])[1], axis=1 )

                newLayer.setWeights( finalWMatrix )
                newLayer.setBias( finalBMatrix )

                newLayer = newNetwork.mutate(newLayer)
                newNetwork.addLayer(newLayer)
            else:
                newNetwork.addLayer(v1)
        
        return newNetwork

    def mutate(self, layer):
        if(isinstance(layer, FullyConnectedLayer)):
            weights = layer.getWeights()
            bias = layer.getBias()

            wShape = np.shape(weights)

            for i in range(0, wShape[0]):
                for j in range(0, wShape[1]):
                    r = random()
                    if(r <= self.__mutationP):
                        r = random()
                        if(r <= 0.5):
                            weights[i, j] -= self.__mutationIntensity
                        else:
                            weights[i, j] += self.__mutationIntensity

            for i in range(0, np.shape(bias)[0]):
                for j in range(0, np.shape(bias)[1]):
                    r = random()
                    if(r <= self.__mutationP):
                        r = random()
                        if(r <= 0.5):
                            bias[i, j] -= self.__mutationIntensity
                        else:
                            bias[i, j] += self.__mutationIntensity
            
            nLayer = FullyConnectedLayer(wShape[0], wShape[1])
            nLayer.setWeights(weights)
            nLayer.setBias(bias)
            return nLayer

        return layer
