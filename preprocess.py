import cv2
from skimage.morphology import thin, skeletonize
from skimage.util import invert
import numpy as np
from matplotlib import pyplot as plt

def input_image(str):
    img = cv2.imread(str)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def filter_image(img):
    r, c = img.shape
    for i in range(0,r):
        for j in range(0,c):
            if img[i, j] <= 30:
                img[i, j] = 0
    return img

def binarize_img(img):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def otsu_thresh(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def gaussian_blur(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return  blur

def thin_image(img):
    img = otsu_thresh(img)
    img = invert(img)
    img[img == 255] = 1
    thinn = thin(img)
    return thinn

def cut_image(img): #expects a binarized image
    minx=img.shape[0]
    miny=img.shape[1]
    maxx=0
    maxy=0

    img = otsu_thresh(img)
    img = abs(255-img)
    hsum = np.sum(img, axis=1)
    for i in range(0, hsum.shape[0]):
        if(hsum[i] != 0):
            minx = min(minx, i)
            maxx = max(maxx, i)

    # print(hsum)
    vsum = np.sum(img, axis=0)
    for i in range(0, vsum.shape[0]):
        if(vsum[i] != 0):
            miny = min(miny, i)
            maxy = max(maxy, i)
    # print(vsum)

    # print(minx, maxx, miny, maxy)
    if(minx-2 >=0):
        minx = minx-2
    elif (minx - 1 >= 0):
        minx = minx - 1
    if(maxx+2 < img.shape[0]):
        maxx = maxx+2
    elif (maxx + 1 < img.shape[0]):
        maxx = maxx + 1
    if (miny - 2 >= 0):
        miny = miny - 2
    elif (miny - 1 >= 0):
        miny = miny - 1
    if (maxy + 2 < img.shape[1]):
        maxy = maxy + 2
    elif (maxy + 1 < img.shape[1]):
        maxy = maxy + 1

    img = img[minx:maxx+1, miny:maxy+1]
    img = abs(255 - img)
    return img