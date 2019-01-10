# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:37:24 2018

@author: wilso
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np 
from skimage.transform import resize
import scipy


with open("original_files/X_test_221.pickle",'rb') as fp: 
    X_original = pickle.load(fp)

with open("original_files/y_test_221.pickle",'rb') as fp: 
    keypoints_original = pickle.load(fp)

        
X_original = np.array(X_original)

keypoints_original = np.array(keypoints_original)

#Transform images and keypoints to the same size
X = [] 
y = []

for i in range(np.shape(X_original)[0]):
    rows, columns, channels = np.shape(X_original[i])
    x1 = rows / 1536
    x2 = columns / 2048
    aux = np.array(X_original[i])
    aux = resize(aux, (aux.shape[0] / x1, aux.shape[1] / x2))
    aux = scipy.misc.imresize(aux,25)
    aux = np.reshape(aux,(384,512,3))
    X.append(aux)
    for j in range(np.shape(keypoints_original)[1]): 
        if(j % 2 == 0):
            keypoints_original[i][j] /= x2
        else: 
            keypoints_original[i][j] /= x1
    y.append(keypoints_original[i] * 0.25)    

    
print(np.shape(X))
print(np.shape(y))

X = np.array(X)

print(np.max(X),'max')
y = np.array(y)


x_s = []
y_s = [] 
  
for i in range(X.shape[0]):
    x_s.append(y[i][0:74:2])
    y_s.append(y[i][1:75:2])     
    plt.imshow(X[i], cmap='gray')
    plt.plot(x_s[i],y_s[i],'o')
    plt.show() 
    #fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    

with open("processed_files/X_test_221.pickle", "wb") as output_file:
    pickle.dump(X,output_file)
    
with open("processed_files/y_test_221.pickle", "wb") as output_file:
    pickle.dump(y,output_file)
    
    
