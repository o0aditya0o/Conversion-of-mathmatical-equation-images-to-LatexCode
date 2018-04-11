import os
import utility as util
import preprocess as pre
import cv2
import pickle
import features as fea
import mapping as mp
import numpy as np

'''
Heuristics:

1. Operator can never be superscript or subscript
2. operator can never have superscript or subscript

'''
def inside(a,b):
    if a[0] <= b[0] and a[0]+a[2] >= b[0]+b[2] and a[1] <= b[1] and a[1]+a[3] >= b[1]+b[3]:
        return 1
    return 0


def overlap(a,b):
    if a[0] <= b[0] and (a[0] + a[2]) >= b[0]:
        return 1
    return 0


def combine(a,b):
    ans = [0,0,0,0]
    ans[0] = min(a[0], b[0])
    ans[1] = min(a[1], b[1])
    x = max(a[0] + a[2], b[0] + b[2])
    y = max(a[1] + a[3], b[1] + b[3])
    ans[2] = x - ans[0]
    ans[3] = y - ans[1]
    return ans


def align(img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    a = []
    for c in contours:
        temp = []
        x,y,w,h = cv2.boundingRect(c)
        temp.append(x)
        temp.append(y)
        temp.append(w)
        temp.append(h)
        a.append(temp)

    a.sort()
    a.pop(0)
    fin = []
    idx = 0
    prev = 0
    for b in a:
        if prev == 1:
            prev = 0
            idx = idx + 1
            continue
        flag = 1
        for g in a:
            if inside(g, b) and b != g:
                flag = 0
                break
        if flag == 1:
            id = 0
            for h in a:
                if id == idx + 1:
                    if overlap(b, h):
                        b = combine(b, h)
                        prev = 1
                    break
                id = id + 1
            fin.append(b)
        idx = idx + 1

    align = []
    px1 = px2 = -1
    py1 = py2 = -1
    idx1 = 0
    prev = 0
    # print(fin)
    with open('MLPClassifier.pkl', 'rb') as f:
        clf1 = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    for c in fin:
        x1 = c[0]
        y1 = c[1]
        x2 = x1 + c[2]
        y2 = y1 + c[3]
        img_crop = img[y1:y2, x1:x2]
        feat = fea.get_data(img_crop)
        temp = []
        temp.append(feat)
        temp = scaler.transform(temp)
        y = clf1.predict(temp)
        y = mp.list[int(y[0])]
        operators = ['≤','≥','≠','÷','×','±' ,'∑','∫','=','+','-','/','*']
        is_prev_operator = 0
        if idx1 == 0 or (y in operators):
            align.append(0)
            prev = 0
            is_prev_operator = 1
        elif is_prev_operator == 1:
            align.append(0)
            is_prev_operator = 0
        elif px2 < x1:
            if (py1 + py2)/2 > y2:
                if prev == -1:
                    align.append(0)
                    prev = 0
                else:
                    align.append(1)
                    prev = 1
            elif (py1 + py2)/2 < y1:
                if prev == 1:
                    align.append(0)
                    prev = 0
                else:
                    align.append(-1)
                    prev = -1
            else:
                align.append(prev)
            is_prev_operator = 0
        px1 = x1
        px2 = x2
        py1 = y1
        py2 = y2
        idx1 = idx1 + 1
    # print(align)
    return align