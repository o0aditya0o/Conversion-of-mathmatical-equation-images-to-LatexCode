import cv2
import os
import preprocess as pre
import utility as util
import segmentation as seg
import features as fea
import pickle
import mapping as mp
import tolatex as tx
import doit as dt
import alignmentchecker as ali
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import numpy as np


def numhorsplit( img ):
    img = np.rot90(img, 1)
    # vertical histogram
    img_not = abs(255 - img)
    histo = img_not.sum(axis=0)

    # smoothening histogram for better detection of words
    if (len(histo) < 7):
        # print("got it")
        l = []
        l.append(img)
        return l
    y_smooth = savgol_filter(histo, 7, 3)

    minimas = argrelextrema(y_smooth, np.less)
    arr = minimas[0]
    # remooving unwanted minimas
    minn = []
    minn.append(0)
    for i in arr:
        if (histo[i] == 0):
            minn.append(i)
        else:
            for j in range(i - 5, i + 6):
                if (j < 0):
                    continue
                if (histo[j] == 0):
                    minn.append(j)
                    break
    minn.append(len(histo) - 1)
    # print(minn)
    clist = []
    alpha_count = 0;
    prev = -1
    # util.display_image(img)
    for i in minn:
        if (prev != -1):
            ch = img[:, prev:i + 1]
            ch_not = abs(255 - ch)
            if (np.sum(ch_not) >= 6 * 255):
                alpha_count = alpha_count + 1
        prev = i
    return alpha_count

def split(img):
    img = np.rot90(img, 1)
    # vertical histogram
    img_not = abs(255 - img)
    histo = img_not.sum(axis=0)

    # smoothening histogram for better detection of words
    if (len(histo) < 7):
        # print("got it")
        l = []
        l.append(img)
        return l
    y_smooth = savgol_filter(histo, 7, 3)

    minimas = argrelextrema(y_smooth, np.less)
    arr = minimas[0]
    # remooving unwanted minimas
    minn = []
    minn.append(0)
    for i in arr:
        if (histo[i] == 0):
            minn.append(i)
        else:
            for j in range(i - 5, i + 6):
                if (j < 0):
                    continue
                if (histo[j] == 0):
                    minn.append(j)
                    break
    minn.append(len(histo) - 1)
    # print(minn)
    clist = []
    alpha_count = 0;
    prev = -1
    # util.display_image(img)
    for i in minn:
        if (prev != -1):
            ch = img[:, prev:i + 1]
            ch_not = abs(255 - ch)
            if (np.sum(ch_not) >= 6 * 255):
                ch = np.rot90(ch, 3)
                clist.append(ch)
        prev = i
    return clist


def recurse(img):
    img_not = abs(255 - img)
    # vertical histogram
    histo = img_not.sum(axis=0)

    # smoothening histogram for better detection of words
    if (len(histo) < 7):
        # print("got it")
        l = []
        l.append(img)
        return l
    y_smooth = savgol_filter(histo, 7, 3)

    # comparing histogram after smoothening
    # plt.plot(histo)
    # plt.plot(y_smooth)

    minimas = argrelextrema(y_smooth, np.less)
    arr = minimas[0]
    # print(arr)
    # print(arr)
    # remooving unwanted minimas
    minn = []
    minn.append(0)
    for i in arr:
        if (histo[i] == 0):
            minn.append(i)
        else:
            for j in range(i - 5, i + 6):
                if (j < 0):
                    continue
                if (histo[j] == 0):
                    minn.append(j)
                    break

    # print(minn)
    minn.append(len(histo) - 1)

    # creating a list of chars
    clist = []
    prev = -1
    previ = -1
    res = ""
    flag =0
    # print("lol")
    # print(minn)
    with open('MLPClassifier.pkl', 'rb') as f:
        clf1 = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    for i in minn:
        flag=0
        if (previ != -1):
            ch = img[:, previ:(i+1)]
            ch_not = abs(255 - ch)
            if (np.sum(ch_not) >= 6 * 255):
                print("number of splits " + str(numhorsplit(ch)))
                # xlist = split(ch)
                # for v in xlist:
                #     util.display_image(v)
                # if numhorsplit(ch) == 3:
                #     flag = 1
                #     res = res + dt.give_me_the_equation(img[:, prev: previ ])
                #     prev = i+1
                #     clist = split(ch)
                #     if(1): #or clist[1] == fraction
                #         res = res + "\\frac{"
                #         res = res + recurse(clist[0])+ "}{" + recurse(clist[2]) + "}"
                if numhorsplit(ch) > 2:
                    xlist = split(ch)
                    idx = 0
                    minus_pos = 0
                    char_operator = '-'
                    for v in xlist:
                        x = fea.get_data(v)
                        temp = []
                        temp.append(x)
                        temp = scaler.transform(temp)
                        y = clf1.predict(temp)
                        y = mp.list[int(y[0])]
                        print(y)
                        if y == '-' or y == '∑' or y == '∫':
                            print("the index here is " + str(idx))
                            util.display_image(v)
                            minus_pos = idx
                            char_operator = y
                        idx = idx + 1
                    num = xlist[0]
                    print("minus " + str(minus_pos))
                    den = xlist[minus_pos+1]
                    util.display_image(den)
                    idx = 0
                    for v in xlist:
                        if idx >= minus_pos:
                            break
                        if idx != 0:
                            num = np.concatenate((num, v), axis = 0)
                        idx = idx + 1
                    idx = 0
                    for v in xlist:
                        if idx <= minus_pos:
                            idx = idx + 1
                            continue
                        if idx != minus_pos+1:
                            den = np.concatenate((den, v), axis = 0)
                        idx = idx + 1
                    res = res + dt.give_me_the_equation(img[:, prev: previ])
                    prev = i + 1
                    util.display_image(num)
                    util.display_image(den)
                    if char_operator == '-':  # or clist[1] == fraction
                        res = res + "\\frac{"
                        res = res + recurse(num) + "}{" + recurse(den) + "}"
                    elif char_operator == '∫':
                        res = res + "\\int_{"
                        res = res + recurse(num) + "}^{" + recurse(den) + "}"
                    else:
                        res = res + "\\sum_{"
                        res = res + recurse(num) + "}^{" + recurse(den) + "}"

        previ = i
        if prev == -1:
            prev = i
    if flag == 0:
        res = res + dt.give_me_the_equation(img[:, prev:previ + 1])
    return res

# input the image
path = os.getcwd() + '\\TestEquations\\' + 'eq18.jpg'
# print(path)
img = pre.input_image(path)
util.display_image(img)
# Converting light dark pixels ( <= 30 ) to black ( = 0 )
img = pre.filter_image(img)
# Thresholding OTSU

img = pre.otsu_thresh(img)
align = ali.align(img)
print(recurse(img))
# util.display_image(img)
