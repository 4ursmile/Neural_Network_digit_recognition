
from tkinter import W
import numpy as np
import pandas as pd
import time
import os
loadData = pd.read_csv('train/train.csv')
data = np.array(loadData)
m,n = data.shape
np.random.shuffle(data) # we dont know about data so we want to shuffle it randomly 

data_Train = data[: m].T #data (m,n) so we want it to (n,m)
Y_Train = data_Train[0] #first row is out Y label
X_Train = data_Train[1:n] # 1 to n is data image
X_Train = X_Train/255 # force if to [0,1]

def Init_Pram():
  W1 = np.random.rand(10, n-1)  - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  #We have 10 neural in L2 is our output layer
  W2 = np.random.rand(10,10) - 0.5
  b2 = np.random.rand(10,1) - 0.5
  return W1, b1, W2, b2
def Reuse_Pram():
  W1 = np.load('AfterTrainData/W1trained.npy')
  W2 = np.load('AfterTrainData/W2trained.npy')
  b1 = np.load('AfterTrainData/b1trained.npy')
  b2 = np.load('AfterTrainData/b2trained.npy')
  return W1, b1, W2, b2
def ReLU(Z):
  return np.maximum(0,Z)
def Signoid(Z):
  return 1/(1+np.exp(-Z))
def SoftMax(Z):
  exp = np.exp(Z - np.max(Z))
  return exp / exp.sum(axis=0)

def  Forward_Propagation(X, W1, b1, W2, b2):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = SoftMax(Z2)
  return Z1, A1, Z2, A2

def init_Y(Y):
  # Create a Matrix(Y.size(), Y.max() +1) with all values are zero in this case we have recognize 10 digits(from 0 to 9) so max must be 9 + 1 we have 10 colum
  aY = np.zeros((Y.size, Y.max() + 1)) 
  aY[np.arange(Y.size), Y] = 1 # Assign right number in each dataset is 1
  aY = aY.T
  return aY
def sign_ReLU(Z):
  return Z>0
def Back_Propagation(X,Y,Z1,A1,Z2,A2,W1,W2):
  preY = init_Y(Y)
  dZ2 = A2- preY
  dW2 = 1/m * dZ2.dot(A1.T)
  db2 = 1/m * np.sum(dZ2,1)

  dZ1 = W2.T.dot(dZ2) * sign_ReLU(Z1)
  dW1 = 1/m * dZ1.dot(X.T)
  db1 = 1/m * np.sum(dZ1,1)
  return dW1, db1, dW2, db2
def Update_Parameter(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha*dW1
  b1 = b1 - alpha*db1.reshape(10,1)
  W2 = W2 - alpha*dW2
  b2 = b2 -  alpha*db2.reshape(10,1)
  return W1, b1, W2, b2
#Two function below to calculate Accurancy of your result (A / Y (input data))
def Get_Prediction(A):
  return np.argmax(A,0)
def Get_Accuracy(prediction, Y):
  print(prediction,Y)
  accuracy = np.sum(prediction == Y) / Y.size
  return accuracy
def Gradient_Descent(X, Y, alpha, number_Step, flag):
  if (flag):
    W1,b1,W2,b2 = Reuse_Pram()
  else:
    W1,b1,W2,b2 = Init_Pram()
  for i in range(number_Step):
    Z1, A1, Z2, A2 = Forward_Propagation(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = Back_Propagation(X,Y,Z1,A1,Z2,A2,W1,W2)
    W1, b1, W2, b2 = Update_Parameter(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
    if (i%10 == 0):
      print("step: ", i)
      prediction =  Get_Prediction(A2)
      print(Get_Accuracy(prediction,Y))
  return W1,b1, W2, b2

alpha = 0.1
nStep = 2000
time_Log = 0
tic = time.time()  
if (os.path.exists('AfterTrainData/W1trained.npy') == 0):
  W1, b1, W2, b2 = Gradient_Descent(X_Train, Y_Train,  alpha, nStep, 0)
else:
  W1, b1, W2, b2 = Gradient_Descent(X_Train, Y_Train,  alpha, nStep, 1)


toc = time.time()
time_Log = toc-tic
print(time_Log)
#Save it
np.save('AfterTrainData/W1trained', W1)
np.save('AfterTrainData/b1trained', b1)
np.save('AfterTrainData/W2trained', W2)
np.save('AfterTrainData/b2trained', b2)
_, _, _, A2 = Forward_Propagation(X_Train, W1, b1, W2, b2)
predictions =  np.argmax(A2,0)
accuracy_Log = np.sum(predictions==Y_Train)/Y_Train.size
#Write TrainLog
import os
f = os.__file__
if (os.path.exists('TrainLog.txt')):
  f = open("TrainLog.txt", "a")
else:
  f =  open("TrainLog.txt", "x")
f.write("\n\nLearning rate: " + str(alpha))
f.write("\nNumber of iterations: "+ str(nStep))
f.write("\nTime to train is: " + str(time_Log))
f.write("\nAccuracy on training set: "+ str(accuracy_Log))
f.close()