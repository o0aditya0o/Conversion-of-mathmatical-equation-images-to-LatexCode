import os
import utility as util
import preprocess as pre
import cv2
import numpy as np

# input the image
path = os.getcwd() + '\\TestEquations\\' + 'eq04.jpg'
print(path)
im = cv2.imread(path,0)
imgray = im
ret,thresh = cv2.threshold(imgray,127,255,0)
util.display_image(im)
org = thresh
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


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

# print(contours)
# cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
# cv2.imshow("Keypoints", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(len(contours))
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
        if inside(g,b) and b!=g:
            flag = 0
            break
    if flag == 1:
        id = 0
        f2 = 0
        for h in a:
            if id == idx+1:
                if overlap(b,h):
                    print(b)
                    print(h)
                    print(overlap(b,h))
                    print("------------------")
                    b = combine(b,h)
                    prev = 1
                break
            id = id + 1
        fin.append(b)
    idx = idx + 1

print(fin)

align = []
px1 = px2 = -1
py1 = py2 = -1
idx1 = 0
prev = 0

for c in fin:
    x = c[0]
    y = c[1]
    w = c[2]
    h = c[3]
    cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Bounding Box", org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
for c in fin:
    print("Here")
    x1 = c[0]
    y1 = c[1]
    x2 = x1 + c[2]
    y2 = y1 + c[3]
    if idx1 == 0:
        align.append(0)
        prev = 0
    elif px2 < x1:
        print("Inner loop")
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
            align.append(0)
            prev = 0
    px1 = x1
    px2 = x2
    py1 = y1
    py2 = y2
    idx1 = idx1 + 1
    print(align)

print(align)