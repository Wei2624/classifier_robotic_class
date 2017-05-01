import cv2
import os
from os import listdir
import numpy as np
import scipy
import sys
from sklearn.decomposition import PCA
import csv
from sklearn.externals import joblib
from sklearn import svm

pca_model = joblib.load("pca.pkl")
model = joblib.load("classifier.pkl")


def classifier(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_img = cv2.resize(gray_image,(140,140),interpolation=cv2.INTER_AREA)
    input_img = input_img.reshape((19600))
    input_img = pca_model.transform(input_img)
    result = model.predict(input_img)
    print result
    return int(result[0])



test = cv2.imread('test.jpg',-1)

print classifier(test)
