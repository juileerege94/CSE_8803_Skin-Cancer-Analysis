import numpy as np
from sklearn.cluster import KMeans
import cv2
import scipy.ndimage as ndimage
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import glob

################### applying CLAHE, median blur and gaussian blurring to images

for filename in glob.glob('../ISIC_cleaned_data/*.jpg'):

    img = cv2.imread(filename)
    img = cv2.resize(img ,(256, 256))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(labimg)

    new_l = clahe.apply(l)
    newimg = cv2.merge((new_l, a, b))

    smoothed = cv2.medianBlur(new_l, 11)
    block = 64
    local_mean = np.zeros((smoothed.shape[0], smoothed.shape[1]))

    i = 0
    j = 0
    rect = smoothed[i:i+block,j:j+block]
    l_mean = np.mean(rect)

    while(i<255):
        j = 0
        while(j<255):
            rect = smoothed[i:i+block,j:j+block]
            l_mean = np.mean(rect)
            local_mean[i:i+block,j:j+block] = l_mean
            j+=block
        i+=block
    i = 0
    j = 0


    total_mean = np.mean(smoothed)
    residual = np.subtract(smoothed,local_mean)

    nos_img = np.add(residual, total_mean)
    res = np.array(nos_img, dtype=np.uint8)
    ret2,th2 = cv2.threshold(smoothed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    smoothed[smoothed > ret2] = 255
    smoothed[smoothed <= ret2] = 0

    kernel = np.ones((9,9),np.uint8)

    des = cv2.bitwise_not(smoothed)
    des = cv2.morphologyEx(des, cv2.MORPH_CLOSE, kernel)
    f_name = '../ISIC_Contours_localglobal/' + filename[21:]
    cv2.imwrite(f_name, des)
