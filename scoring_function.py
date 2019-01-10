# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:10:59 2018

@author: wilso
"""


from shapely.geometry import LineString, Point
import numpy as np
import scipy.interpolate as interpolate
import pickle 
import sys
import os

diagonal_dataset120 = 2549.7938
diagonal_dataset221 = 2701.6085

def compute_euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))
    
def get_curves_distance(points_a,points_b,n_points=5000):
    points_a = np.asarray(points_a).reshape([-1,2])
    points_b = np.asarray(points_b).reshape([-1,2])

    distance = curve_dist_aux(points_a,points_b,n_points)
    distance+= curve_dist_aux(points_b,points_a,n_points)
    distance/=2
    return distance

def curve_dist_aux(curve_points,points,n_points=5000):
    curve = spline(curve_points,n_points)
    curve = np.stack(curve,axis=1)
    curve = LineString(curve)
    distance = 0
    for point in points:
        distance+=curve.distance(Point(point))
    distance/=len(points)
    return distance

def spline(points,n_points=5000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

def scoring(predictions, y, img_shape, dataset="120"):
    
    if dataset == "120":
        normalize = diagonal_dataset120
    elif dataset == "221":
        normalize = diagonal_dataset221

    score = []
    
    diagonal = img_shape[0:2]
    diagonal = np.sqrt(diagonal[0]**2+diagonal[1]**2)
    
    truth_lp = y[0:2]
    truth_midl = y[32:34]
    truth_midr = y[66:68]
    truth_rp = y[34:36]
    
    
    truth_l_breast = y[0:34]
    truth_r_breast = y[34:68]
    
    truth_l_nipple = y[70:72]
    truth_r_nipple = y[72:74]

    score.append(compute_euclidean_distance(predictions[0], truth_lp) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[1], truth_midl) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[2], truth_midr) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[3], truth_rp) * (normalize / diagonal))    

    score.append(get_curves_distance(predictions[4], truth_l_breast) * (normalize / diagonal))
    score.append(get_curves_distance(predictions[5], truth_r_breast) * (normalize / diagonal))

    score.append(compute_euclidean_distance(predictions[6], truth_l_nipple) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[7], truth_r_nipple) * (normalize / diagonal))
    return score

def dense_scoring(scores):
    
    scores = np.asarray(scores)
    end_point_scores = scores[:,0:4]
    end_points = [np.mean(end_point_scores),
                  np.std(end_point_scores),
                  np.max(end_point_scores)
                  ]
    breast_contour_scores = scores[:,4:6]
    breast_contour = [np.mean(breast_contour_scores),
                      np.std(breast_contour_scores),
                      np.max(breast_contour_scores)
                      ]
    
    nipple_scores = scores[:,6:8]
    nipple = [np.mean(nipple_scores),
              np.std(nipple_scores),
              np.max(nipple_scores)
              ]
    
    return end_points, breast_contour, nipple

def to_strange_fmt(y):
    y = np.asarray(y)
    lpredictions = []
    lpredictions.append(y[0:2])
    lpredictions.append(y[32:34])
    lpredictions.append(y[66:68])
    lpredictions.append(y[34:36])
    
    lpredictions.append(y[0:34])
    lpredictions.append(y[34:68])
    
    lpredictions.append(y[70:72])
    lpredictions.append(y[72:74])
    return lpredictions


with open("X_test_cv_5.pickle",'rb') as fp: 
        X_test = pickle.load(fp)  

with open("y_test_cv_5.pickle",'rb') as fp: 
        gnd = pickle.load(fp)  
        
with open("preds_cv_5.pickle",'rb') as fp: 
        preds = pickle.load(fp)          


final_scores = [] 

for j in range(5): 
    predictions = [] 
    
    for i in range(np.shape(preds[j])[0]):
        predictions.append(to_strange_fmt(preds[j][i]))
    
    
    scores = [] 
    
    for i in range(np.shape(X_test)[0]): 
        scores.append(scoring(predictions[i], gnd[j][i], X_test[j][i].shape))
        
    print(dense_scoring(scores))
    
    final_scores.append(dense_scoring(scores))
    
 
final_scores = np.array(final_scores)    
    
print(final_scores)


print(final_scores.mean(axis=0))
    

