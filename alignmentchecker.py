import os
import utility as util
import preprocess as pre
import cv2
import pickle
import features as fea
import mapping as mp
import numpy as np
import math
import segmentation as seg
'''
Heuristics:

1. Operator can never be superscript or subscript
2. operator can never have superscript or subscript

'''

def inside(a, b):
    if a[0] <= b[0] and a[0]+a[2] >= b[0]+b[2] and a[1] <= b[1] and a[1]+a[3] >= b[1]+b[3]:
        return 1
    return 0


def overlap(a,b):
    if a[0] <= b[0] and (a[0] + a[2]) >= b[0]:
        return 1
    return 0


def combine(a,b):
    ans = [0, 0, 0, 0]
    ans[0] = min(a[0], b[0])
    ans[1] = min(a[1], b[1])
    x = max(a[0] + a[2], b[0] + b[2])
    y = max(a[1] + a[3], b[1] + b[3])
    ans[2] = x - ans[0]
    ans[3] = y - ans[1]
    return ans

def align(img, model):
    # util.display_image(img)
    # characters = seg.split_characters(img)
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
    if len(a):
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
    pc = '#'
    # print(fin)
    idx_chars = 0
    is_prev_operator = 0

    for c in fin:
        x1 = c[0]
        y1 = c[1]
        x2 = x1 + c[2]
        y2 = y1 + c[3]
        height = (px2 - px1)/3
        img_crop = img[y1:y2, x1:x2]
        # print(img_crop.dtype)
        # zeros = np.zeros((128,128), dtype = np.uint8)
        # startx = math.floor((128 - c[3])/2)
        # starty = math.floor((128 - c[2])/2)
        # zeros[startx:startx+c[3], starty:starty+c[2]] = img_crop
        y = util.predict_class(img_crop, model)
        operators = ['≤','≥','≠','÷','×','±' ,'∑','∫','=','+','/','*']
        if idx1 == 0:
            align.append(0)
            prev = 0
        # elif y == '-':
        #     is_prev_operator = 1
        #     if ((pc not in mp.ascender) and py1 - y2 < height):
        #         align.append(1)
        #         prev = 1
        #     elif ((pc not in mp.descender) and y1 - )
        elif y in operators:
            align.append(0)
            prev = 0
            is_prev_operator = 1
        elif is_prev_operator == 1:
            align.append(prev)
            is_prev_operator = 0
        elif px2 < x1:
            if (((py1 + py2)/2 >= y2) or ((pc not in mp.descender) and py2 - y2 > height) or
                        ((pc in mp.descender) and (y in mp.descender) and py2 - y2 > height)):
                if prev == -1:
                    align.append(0)
                    prev = 0
                else:
                    align.append(1)
                    prev = 1
            elif (((py1 + py2)/2 <= y1) or ((pc not in mp.ascender) and y1 - py1 > height) or
                      ((pc in mp.ascender) and (y in mp.ascender) and y1 - py1 > height)):
                if pc == 'i' and y == 'n':
                    print(str(y1) + ' ' + str(py1) + ' ' + str(pc not in mp.ascender) + ' ' + str(y in mp.ascender))
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
        pc = y
        idx1 = idx1 + 1
        idx_chars = idx_chars + 1
        print(y + ' ' + str(prev))
    # print(align)
    return align