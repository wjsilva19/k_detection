# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:32:44 2018

@author: eduardo
"""

import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.interpolate as interpolate
import scipy.ndimage.morphology as morpho
import features
import scipy.ndimage.morphology as morpho

Y = np.load("datasets/y_train_221.pickle")
X = np.load("datasets/X_train_221.pickle")

def create_breast_mask(shape, points, left = True):
    mask = np.zeros(shape)
    if left:
        points = np.flip(np.asarray(points[0: 34]).reshape([-1, 2]), axis=1)
    else:
        points = np.flip(np.asarray(points[34: 68]).reshape([-1, 2]), axis=1)
    y = np.array(spline(points)).transpose()
    y = y.round().astype(int)
    y[:, 0] = np.clip(y[:, 0], 0, shape[0]-1)
    y[:, 1] = np.clip(y[:, 1], 0, shape[1]-1)
    mask[y[:,0],y[:,1]] = 1
    hor = np.linspace(y[0,0],y[-1,0],1000).round().astype(int)
    ver = np.linspace(y[0,1],y[-1,1],1000).round().astype(int)
    mask[hor, ver] = 1
    return morpho.binary_fill_holes(mask)

def spline(points, n_points=10000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

R = 60
filt2 = np.zeros([1,9])
filt2[0,0] = -1
filt2[0,8]=1
    
import scipy
import shortest_paths
from shapely.geometry import Polygon
import skimage.filters as filters



def circular_path_features(img, point):
    
    polar, mapping = shortest_paths.topolar(img, point, R, ret_mapping=True)
    final = np.abs(scipy.signal.convolve2d(polar,filt2,"valid"))
    final2 = final/final.max()*255
    
    paths = shortest_paths.shortest_paths(final2,beta=0.02)[0]
    indexes = np.nonzero(np.abs(paths[:,0,1]-paths[:,719,1]) <= 1)
    polar_path = paths[indexes[0][0],:,:]
    
    mean_intensity = np.mean(final[polar_path[:,0],polar_path[:,1]])
    
    for point in range(polar_path.shape[0]):
        polar_path[point] = mapping(polar_path[point])
    
    poly = Polygon(polar_path)
    A = poly.area
    P = poly.length
    Sa = (4*np.pi*A)/P**2
    da = np.sqrt((4*A)/np.pi)
    
    return [mean_intensity, Sa, da]



svm_data = dict()

for i in range(104,len(X)):
    print(i)
    svm_data[i] = dict()
    points = np.asarray(Y[i])
    img, points = features.resize(np.average(X[i], axis=2), points)
    
    grad_mag = features.gradient_mag(img)
    maskL = create_breast_mask(img.shape, points, True)
    maskL = morpho.binary_erosion(maskL, iterations=10)
    x, _ = maskL.nonzero()
    maskL[int(x.min()):int(x.min()+(x.max()-x.min())*0.4), :] = 0
    
    maskR = create_breast_mask(img.shape, points, False)
    maskR = morpho.binary_erosion(maskR, iterations=10)
    x, _ = maskR.nonzero()
    maskR[int(x.min()):int(x.min()+(x.max()-x.min())*0.4), :] = 0
    
    harr = features.harris_corner_measure(img)
    corner_infoL = harr * maskL
    corner_infoR = harr * maskR
    x_ = []
    y_ = []
    values_ = []
    
    
    for corner_info in [corner_infoL, corner_infoR]:
        for j in range(10):
            x, y = np.unravel_index(np.argmax(corner_info), corner_info.shape)
            #circular_shortest_path(grad_mag, point)
            x_.append(x)
            y_.append(y)
            values_.append(corner_info[x, y])
            corner_info[x-15:x+15, y-15:y+15] = -np.inf
    
    
    img2 = filters.gaussian(img,3)
    for j in range(len(x_)):
        print(j)
        point = (x_[j],y_[j])
        featu = [values_[j], *circular_path_features(img2, point)]
        svm_data[i][point] = featu

    with open("second_svm_data2.pkl", "wb") as file:
        pkl.dump(svm_data, file)
    
    im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    plt.scatter(y_, x_)
    plt.show()




threshold = 16

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


dataset = []
labels = []

for i in range(0, len(svm_data)):
    img, y = features.resize(X[i], np.asarray(Y[i]))
    left = np.flip(y[70:72].reshape([2]), axis=0)
    right = np.flip(y[72:74].reshape([2]), axis=0)
    for key in svm_data[i].keys():
        dist = min(distance(key, left), distance(key, right))
        dataset.append(svm_data[i][key])      
        if dist<threshold:
            labels.append(1)
        else:
            labels.append(0)


final_svm_data = dataset, labels
with open("second_final_svm_data.pkl", "wb") as file:
    pkl.dump(final_svm_data, file)



"""        
    
import skimage.filters as filters
for i in range(7,84):
    print(i)
    points = np.asarray(Y[i])
    img, points = features.resize(np.average(X[i], axis=2), points)
    #img2 = filters.gaussian(img, sigma=5)
    mask = create_breast_mask(img.shape, points, False)+\
            create_breast_mask(img.shape, points, True)
    mask = morpho.binary_erosion(mask, iterations=10)
    corner_info = features.harris_corner_measure(img) * mask
    print(np.log(corner_info.max()))
    #corner_info = (corner_info > (th * corner_info.max()))
    x, y = np.nonzero(corner_info)
    
    im1 = plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
                 
    im2 = plt.imshow(corner_info>th, cmap=plt.cm.viridis, alpha=.8, interpolation='bilinear')
                 
    #plt.imshow(img)
    #plt.scatter(y, x,c="r", s=2)
    plt.show()
    l=input()
    if l == "stop":
        break



debug one:
    Find that all figures have atleast one nipple
    
    
nipps = []    
for all images:
    inside_nipps = []
    find considered points
    for all considered points:
        show_point and ask it it is true or false
    
    save()
        
    
    
img = X[0]
corner_info = features.harris_corner_measure(img)
corner_info = corner_info > (th * corner_info.max())
x, y = np.nonzero(corner_info)


plt.imshow(img)
plt.scatter(y, x)
plt.show()

"""