from cgi import test
from operator import index
from tkinter import Image, image_names
import numpy as np
import pandas as pd
import cv2 as cv
import os
#Create Folder to classify images
if (os.path.exists('Classify') == 0):
  os.mkdir('Classify')
if (os.path.exists('Classify/WrongPredictions') == 0):
  os.mkdir('Classify/WrongPredictions')
for i in range(10):
   path = 'Classify/Number_' + str(i)
   if (os.path.exists(path) == 0):
    os.mkdir(path)

test_Data =  pd.read_csv('train/test.csv')
data = np.array(test_Data)

m,n = data.shape
data = data.T
X = data
X = X/255
def ReLU(Z):
  return np.maximum(0,Z)
def SoftMax(Z):
  exp = np.exp(Z - np.max(Z))
  return exp / exp.sum(axis=0)
def  Forward_Propagation(X, W1, b1, W2, b2):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = SoftMax(Z2)
  return Z1, A1, Z2, A2
W1 = np.load('AfterTrainData/W1trained.npy')
W2 = np.load('AfterTrainData/W2trained.npy')
b1 = np.load('AfterTrainData/b1trained.npy')
b2 = np.load('AfterTrainData/b2trained.npy')
def Make_Predictions(X,W1,b1,W2,b2):
    _,_,_,A2 = Forward_Propagation(X,W1,b1,W2,b2)
    predictions = np.argmax(A2,0)
    return predictions
def Classify_Image(Index, W1,b1, W2, b2):
  img = X[:,Index,None]
  prediction = Make_Predictions(img,W1,b1,W2,b2)
  img = img.reshape(28,28) * 255
  numberPrediction = int(prediction)
  image_Path = 'Classify/Number_' + str(numberPrediction)
  image_Name = '/img' + str(Index) + 'th.jpg'
  cv.imwrite(image_Path + image_Name, img)

  #predictions = Make_Predictions(img, W1,b1,W2,b2)
  #img = img.reshape(28,28)*255
  #print(predictions)
for i in range(m):
  Classify_Image(i, W1, b1, W2, b2)
  print(i)





