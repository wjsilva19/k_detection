# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:04:09 2018

@author: eduardo
"""
import os
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import scipy.ndimage.morphology as morpho
import skimage
import skimage.filters as filters
import scipy.interpolate as interpolate
import cv2
import scipy
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from shapely.geometry import Polygon

import DijkstraPaths
import shortest_paths as sp

"""
_____________________________ MAIN FUNCTIONS __________________________________
"""

# ALGORITHM PARAMETERS AND MODELS
HEIGHT = 972
R = 60
filt2 = np.zeros([1, 9])
filt2[0, 0] = -1
filt2[0, 8] = 1
TH = 0.85*(HEIGHT)//2*(sp.alpha*np.exp(255*sp.beta)+sp.delta)
SEL_PATH_P = (11, 1)
# load the prior model points
file = "breast_contour_prior221.pkl"   
if os.path.isfile(file):
    with open(file, "rb") as file:
        # If these parameters have not been saved please run 
        # save_breast_contour_params
        PRIOR_POINTS = pkl.load(file)

file = "normalize_params221.pkl"
if os.path.isfile(file):
    with open(file, "rb") as file:
        # If these parameters have not been saved please run 
        # save_trained_svm
        norm_mean, norm_std = pkl.load(file)

file = "nipple_classifier221.pkl"
if os.path.isfile(file):
    with open(file, "rb") as file:
        # If this model has not been saved please run 
        # save_trained_svm
        nipple_classifier = pkl.load(file)


def imresize_and_scalling(img, points=None, size=HEIGHT):
    scalling_factor = HEIGHT/img.shape[0]
    img = cv2.resize(img, dsize=(0, 0), fx=scalling_factor, fy=scalling_factor)
    if points is not None:
        points *= scalling_factor
        return img, points
    return img, scalling_factor


def find_breast_endpoints(g_mag):

    # Obtain the bottom half of the image
    offset = g_mag.shape[0]//2
    halg_g_mag = g_mag[offset::, :]

    # Obtain the shortest path in both up and down directions
    paths_up, dists_up = sp.shortest_paths(halg_g_mag)
    paths_down, dists_down = sp.shortest_paths(halg_g_mag[::-1])
    paths_down[:, :, 1] = paths_down[:, ::-1, 1]

    # Find candidates for the patient's body edge
    indexes = sp.select_equal_paths(paths_up, paths_down)
    paths = paths_up[indexes]
    dists = dists_up[indexes]
    paths = paths[dists < TH]
    dists = dists[dists < TH]

    # Select the correct candidate for the left and right edge
    possible_points_to_start = paths[:, -1, 1]
    mid = g_mag.shape[1]/2
    selected = np.abs(possible_points_to_start-mid) > (g_mag.shape[1]*0.15)
    p = possible_points_to_start[selected]
    lp = np.max(p[(p-mid) < 0])
    rp = np.min(p[(p-mid) > 0])
    mid_height = HEIGHT//2
    lp = np.array([mid_height, lp])
    rp = np.array([mid_height, rp])

    # move upwards untill the gradient is small (nipple end-point)
    matrix = sp.construct_distance_matrix(g_mag, lp)
    paths = sp.find_all_paths(matrix, g_mag, lp)
    lp = sp.select_path(paths, g_mag, SEL_PATH_P[0], SEL_PATH_P[1])[0]
    lp = np.asarray(lp)
    matrix = sp.construct_distance_matrix(g_mag, rp)
    paths = sp.find_all_paths(matrix, g_mag, rp)
    rp = sp.select_path(paths, g_mag, SEL_PATH_P[0], SEL_PATH_P[1])[0]
    rp = np.asarray(rp)

    # Mid point is obtained as the mean between left and right points
    mp = (lp+rp)/2
    return lp, mp, rp


def find_breast_contour(g_mag, p1, p2):
    # Adjust the contour prior to the position of this specific image
    prior_points = DijkstraPaths.adjust(PRIOR_POINTS, p1, p2)
    prior_points[:, 0] = np.clip(prior_points[:, 0], 0, g_mag.shape[0]-1)
    prior_points[:, 1] = np.clip(prior_points[:, 1], 0, g_mag.shape[1]-1)

    # compute the prior mask
    prior = np.zeros(g_mag.shape)
    prior[prior_points[:, 0].astype(int), prior_points[:, 1].astype(int)] = 1
    prior = morpho.binary_fill_holes(prior)
    prior = morpho.distance_transform_edt(1-prior)

    # compute the shortest path
    distance_funcs = g_mag, prior
    G = DijkstraPaths.build_graph(distance_funcs, [0, 0, *prior.shape])
    path = DijkstraPaths.shortestPath(G, tuple(p1.astype(int)),
                                      tuple(p2.astype(int)))
    path = np.asarray(path)
    return path


def find_nipple_pos(img, breast_contour):
    # find the points inside the breast where the nipple could be
    mask = create_breast_mask(img.shape[0:2], breast_contour, True)
    mask = morpho.binary_erosion(mask, iterations=10)
    x, _ = mask.nonzero()
    mask[int(x.min()):int(x.min()+(x.max()-x.min())*0.4), :] = 0

    # find nipple candidates
    harr = harris_corner_measure(img)
    corner_info = harr * mask
    x_, y_ = [], []
    values_ = []
    for j in range(10):
        x, y = np.unravel_index(np.argmax(corner_info), corner_info.shape)
        x_.append(x)
        y_.append(y)
        values_.append(corner_info[x, y])
        corner_info[x-15:x+15, y-15:y+15] = -np.inf

    # compute features for nipple candidates
    featu = []
    img2 = filters.gaussian(np.average(img, axis=2), 3)
    for j in range(len(x_)):
        point = (x_[j], y_[j])
        featu.append([values_[j], *circular_path_features(img2, point)])

    # select the most probable candidate
    featu = (np.asarray(featu)-norm_mean) / norm_std
    probs = nipple_classifier.predict_proba(featu)
    nip_index = np.argmax(probs[:, 1])
    return np.asarray((x_[nip_index], y_[nip_index]))


"""
_____________________________ AUXILIARY FUNCTIONS _____________________________
"""


def gradient_mag(img):
    # Returns the gradient magnitude of the input image
    gx = skimage.filters.sobel_h(img)
    gy = skimage.filters.sobel_v(img)
    magnitude = np.sqrt(gx**2+gy**2)
    return magnitude


def harris_corner_measure(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 5, 3, 0.04)
    return dst


def create_breast_mask(shape, points, left=True):
    mask = np.zeros(shape)
    y = np.array(spline(points)).transpose()
    y = y.round().astype(int)
    y[:, 0] = np.clip(y[:, 0], 0, shape[0]-1)
    y[:, 1] = np.clip(y[:, 1], 0, shape[1]-1)
    mask[y[:, 0], y[:, 1]] = 1
    hor = np.linspace(y[0, 0], y[-1, 0], 1000).round().astype(int)
    ver = np.linspace(y[0, 1], y[-1, 1], 1000).round().astype(int)
    mask[hor, ver] = 1
    return morpho.binary_fill_holes(mask)


def spline(points, n_points=10000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:, 0]
    y = points[:, 1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out


def circular_path_features(img, point):
    polar, mapping = sp.topolar(img, point, R, ret_mapping=True)
    final = np.abs(scipy.signal.convolve2d(polar, filt2, "valid"))
    final2 = final / final.max() * 255

    paths = sp.shortest_paths(final2, beta=0.02)[0]
    indexes = np.nonzero(np.abs(paths[:, 0, 1]-paths[:, 719, 1]) <= 1)
    polar_path = paths[indexes[0][0], :, :]

    mean_intensity = np.mean(final[polar_path[:, 0], polar_path[:, 1]])

    for point in range(polar_path.shape[0]):
        polar_path[point] = mapping(polar_path[point])

    poly = Polygon(polar_path)
    A = poly.area
    P = poly.length
    Sa = (4*np.pi*A)/P**2
    da = np.sqrt((4*A)/np.pi)

    return [mean_intensity, Sa, da]


def visualize(X, y, preds, number):
    plt.imshow(X)
    truth_lp = y[0:2]
    truth_midl = y[32:34]
    truth_midr = y[66:68]
    truth_rp = y[34:36]

    truth_l_breast = np.asarray(y[0:34]).reshape([-1, 2])
    truth_r_breast = np.asarray(y[34:68]).reshape([-1, 2])

    truth_l_nipple = y[70:72]
    truth_r_nipple = y[72:74]

    plt.scatter(truth_lp[0], truth_lp[1], c="b")
    plt.scatter(truth_midl[0], truth_midl[1], c="b")
    plt.scatter(truth_midr[0], truth_midr[1], c="b")
    plt.scatter(truth_rp[0], truth_rp[1], c="b")

    plt.plot(truth_l_breast[:, 0], truth_l_breast[:, 1], c="b")
    plt.plot(truth_r_breast[:, 0], truth_r_breast[:, 1], c="b")

    plt.scatter(truth_l_nipple[0], truth_l_nipple[1], c="b")
    plt.scatter(truth_r_nipple[0], truth_r_nipple[1], c="b")

    plt.scatter(preds[0][0], preds[0][1], c="r")
    plt.scatter(preds[1][0], preds[1][1], c="r")
    plt.scatter(preds[2][0], preds[2][1], c="r")
    plt.scatter(preds[3][0], preds[3][1], c="r")

    plt.plot(preds[4][:, 0], preds[4][:, 1], c="r")
    plt.plot(preds[5][:, 0], preds[5][:, 1], c="r")

    plt.scatter(preds[6][0], preds[6][1], c="r")
    plt.scatter(preds[7][0], preds[7][1], c="r")

    plt.savefig("validation_"+str(number)+"_points.png")
    plt.clf()


"""
_____________________________ MAIN FUNCTIONS __________________________________
"""

def transform(y):
    y -= y[0]
    angle = np.arctan2(y[-1, 0], y[-1, 1])
    M = cv2.getRotationMatrix2D((0, 0), (angle*180) / np.pi, 1)
    y = np.dot(y, M[0:2, 0:2])
    y /= np.sqrt(np.sum(y[-1]**2))
    return y


def create_mask(all_shapes):
    final_mask = np.zeros([1800, 2100])
    for y in all_shapes:
        y[:, 0] *= (-1)
        y *= 1000
        y += [100, 500]
        y = np.array(spline(y)).transpose()
        for point in y:
            final_mask[tuple(point.astype(int))] = 1

    final_mask = morpho.binary_fill_holes(final_mask)
    b = morpho.binary_erosion(final_mask, structure=np.ones([3, 3]))
    x, y = np.nonzero(np.logical_and(final_mask, np.logical_not(b)))
    points = np.stack((x, y), axis=1).astype(float)
    points -= [100, 500]
    points /= 1000
    return points


def save_breast_contour_params():
    Y = np.load("datasets/y_train_221.pickle")

    all_shapes = list()
    for curr_y in Y:
        for side in ["left", "right"]:
            if side is "left":
                y = curr_y[0:34]
                y = np.reshape(y, [-1, 2])
            else:
                y = curr_y[34:68]
                y = np.reshape(y, [-1, 2])
                y[:, 0] *= (-1)

            y = transform(y)
            all_shapes.append(y)

    points = create_mask(all_shapes)

    with open("breast_contour_prior221.pkl", "wb") as file:
            pkl.dump(points, file)


def resize(img, points=None, size=HEIGHT):
    scalling_factor = HEIGHT/img.shape[0]
    img = cv2.resize(img, dsize=(0, 0), fx=scalling_factor, fy=scalling_factor)
    if points is not None:
        points *= scalling_factor
        return img, points
    return img


def distance(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def save_trained_svm():
    svm_data = dict()
    for i in range(104, len(X)):
        print(i)
        svm_data[i] = dict()
        points = np.asarray(Y[i])
        img, points = resize(np.average(X[i], axis=2), points)

        maskL = create_breast_mask(img.shape, points, True)
        maskL = morpho.binary_erosion(maskL, iterations=10)
        x, _ = maskL.nonzero()
        maskL[int(x.min()):int(x.min()+(x.max()-x.min())*0.4), :] = 0

        maskR = create_breast_mask(img.shape, points, False)
        maskR = morpho.binary_erosion(maskR, iterations=10)
        x, _ = maskR.nonzero()
        maskR[int(x.min()):int(x.min()+(x.max()-x.min())*0.4), :] = 0

        harr = harris_corner_measure(img)
        corner_infoL = harr * maskL
        corner_infoR = harr * maskR
        x_ = []
        y_ = []
        values_ = []

        for corner_info in [corner_infoL, corner_infoR]:
            for j in range(10):
                x, y = np.unravel_index(np.argmax(corner_info),
                                        corner_info.shape)
                x_.append(x)
                y_.append(y)
                values_.append(corner_info[x, y])
                corner_info[x-15:x+15, y-15:y+15] = -np.inf

        img2 = filters.gaussian(img, 3)
        for j in range(len(x_)):
            print(j)
            point = (x_[j], y_[j])
            featu = [values_[j], *circular_path_features(img2, point)]
            svm_data[i][point] = featu

    threshold = 16

    dataset = []
    labels = []

    for i in range(0, len(svm_data)):
        img, y = resize(X[i], np.asarray(Y[i]))
        left = np.flip(y[70:72].reshape([2]), axis=0)
        right = np.flip(y[72:74].reshape([2]), axis=0)
        for key in svm_data[i].keys():
            dist = min(distance(key, left), distance(key, right))
            dataset.append(svm_data[i][key])
            if dist < threshold:
                labels.append(1)
            else:
                labels.append(0)

    dataset = np.asarray(dataset)
    labels = np.asarray(labels)

    mean, std = dataset.mean(axis=0), dataset.std(axis=0)
    dataset -= mean
    dataset /= std

    normalize_params = mean, std
    with open("normalize_params221.pkl", "wb") as file:
        pkl.dump(normalize_params, file)

    param_grid = [
      {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
       'kernel': ['linear']},
      {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
       'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
       'kernel': ['rbf']}]

    svc = SVC(probability=True, class_weight={0: 0.183928, 1: 0.816071})
    clf = GridSearchCV(svc, param_grid, verbose=2)
    clf.fit(dataset, labels)
    results = clf.cv_results_
    test_score = results["mean_test_score"]
    params = results["params"]

    classifier = SVC(probability=True, class_weight={0: 0.183928, 1: 0.816071},
                     **params[np.argmax(test_score)])

    classifier.fit(dataset, labels)

    with open("nipple_classifier221.pkl", "wb") as file:
        pkl.dump(classifier, file)


"""
_____________________________ MAIN ____________________________________________
"""
if __name__ == "__main__":
    # RUN PARAMETERS
    SHOW_IMAGES = True
    split = "test"  # ["train", "test"]
    dataset = "221"  # ["120", "221"]

    # Load dataset
    X = np.load("datasets/X_"+str(split)+"_"+str(dataset)+".pickle")
    Y = np.load("datasets/y_"+str(split)+"_"+str(dataset)+".pickle")
    N = X.shape[0]

    all_predictions = []

    for i in range(N):  # For each image in the dataset
        print("Current image:", i)

        preds = []
        img, scalling = imresize_and_scalling(X[i])
        img_grey = np.average(img, axis=2)
        # compute gradient magnitude
        g_mag = gradient_mag(img_grey)
        # endpoints: left, mid and right
        lp, mp, rp = find_breast_endpoints(g_mag)

        # save endpoints in predictions list
        preds.append(lp[::-1]/scalling)
        preds.append(mp[::-1]/scalling)
        preds.append(mp[::-1]/scalling)
        preds.append(rp[::-1]/scalling)

        # find the left breast countour
        left_contour = find_breast_contour(g_mag, lp, mp)

        # obtain 17 equaly spaced points (key-points)
        jump = (left_contour.shape[0]-1)/16
        indexes = [int(x*jump) for x in range(17)]
        left_contour = left_contour[indexes, :]

        # find the right breast countour
        right_contour = find_breast_contour(g_mag, mp, rp)

        # obtain 17 equaly spaced points (key-points)
        jump = (right_contour.shape[0]-1)/16
        indexes = [int(x*jump) for x in range(17)]
        right_contour = right_contour[indexes, :]

        # Find the nipple position for each breast
        nip_l = find_nipple_pos(img, left_contour)
        nip_r = find_nipple_pos(img, right_contour)

        # Add breast contour key-points and nipple key-points to pred. list
        preds.append(left_contour[:, ::-1] / scalling)
        preds.append(right_contour[:, ::-1] / scalling)
        preds.append(nip_l[::-1]/scalling)
        preds.append(nip_r[::-1]/scalling)

        if SHOW_IMAGES:
            visualize(X[i], Y[i], preds, i)

        # Add predictions list to the global results
        all_predictions.append(preds)

    # Save results
    filename = "predictions_"+str(split)+"_"++str(dataset)+".pkl"
    with open(filename, "wb") as file:
        pkl.dump(all_predictions, file)
