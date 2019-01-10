import numpy as np 
import csv
import _pickle as cPickle
import matplotlib.pyplot as plt
import math
import cv2
import scipy.misc


"""Function to generate heatmaps"""
def get_pdf(im, kpts, sigma):
    w, h, channels = im.shape
    
    x = np.linspace(0, h-1, h*1)
    y = np.linspace(0, w-1, w*1)
    [XX, YY] =  np.meshgrid(y,x)
    sze = XX.shape[0] * XX.shape[1]
    mvg = np.zeros((sze));    
    std = sigma
    p = 2
    count=0
    for i in range(0,37):
        mu = np.array([kpts[i][1], kpts[i][0]]).reshape((2,1)) 
        mu = np.tile(mu, (1, sze))
        mcov = np.identity(2) * std
        
        X = np.array([np.ravel(XX.T), np.ravel(YY.T)])
        
        temp0 = 1 / ( math.pow(2*math.pi, p/2) * \
                    math.pow(np.linalg.det(mcov), 0.5) )
        
        temp1 = -0.5*(X-mu).T
        temp2 = np.linalg.inv(mcov).dot(X-mu) 
        
        temp3 = temp0 * np.exp(np.sum(temp1 * temp2.T, axis=1))
        maximum = max(temp3.ravel())
        
        mvg = mvg + temp3
        count += 1
    
        mvg[mvg>maximum] = maximum
        
    mvg = mvg.reshape((XX.shape[1], XX.shape[0]))
    
    mvg = ( mvg - min(mvg.ravel()) ) / ( max(mvg.ravel()) - min(mvg.ravel()) )
    
    mvg = mvg * 255.0
    mvg = cv2.resize(mvg, (h, w), interpolation = cv2.INTER_CUBIC)
    mvg = mvg / 255.0
    mvg[mvg<0] = 0
   
    return mvg



"""Convert keypoints to tupple of keypoints"""
def tupple(data):
    
    points = np.array(data)
    
    tupple_keypoints = []
    tupple_aux = [] 
    x = []
    y = []
    
    for i in range(np.shape(points)[0]): 
        for j in range(74): 
            if (j % 2 == 0): 
                x.append(int(points[i][j]))
            else:
                y.append(int(points[i][j]))    
        for z in range(37):
            tupple_aux.append((int(x[z]),int(y[z])))
        tupple_keypoints.append(tupple_aux)
        x = [] 
        y = [] 
        tupple_aux = [] 
    
    for i in range(np.shape(points)[0]):
        tupple_keypoints.append(points[i])
    
    keypoints = np.array(tupple_keypoints)

    return keypoints


"""Heatmaps"""
def heatmap_generation(X,keypoints):

    density_map = []
    
    for i in range(np.shape(X)[0]): 
        oriImg = X[i]
        mapa = get_pdf(oriImg,keypoints[i],400)
        density_map.append(mapa)
        
    density_map = np.array(density_map)
    
    return density_map


""" Data """
with open("processed_files/X_test_221.pickle",'rb') as fp: 
    X = cPickle.load(fp)

with open("processed_files/y_test_221.pickle",'rb') as fp: 
    data = cPickle.load(fp)
    
keypoints = tupple(data)    

density_map = heatmap_generation(X,keypoints)

#print(np.shape(density_map))

#for i in range(X.shape[0]): 
#    plt.imshow(X[i])
#    plt.imshow(density_map[i], interpolation='nearest', alpha=0.6)
#    plt.show()


with open("processed_files/heatmaps_test_221.pickle", "wb") as output_file:
    cPickle.dump(density_map,output_file)



