import preprocess as pre
import utility as util
import numpy as np
import cv2
from skimage.util import invert
from scipy import signal
import math


# from matplotlib import pyplot as plt

def pre_init(img):
    img = pre.cut_image(img)
    if img.shape[0] == 0 or img.shape[1] == 0:
        return -1, -1, -1
    aspect = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
    img = pre.otsu_thresh(img)
    new_img = pre.thin_image(img)

    # changes for end-points counting
    # img = util.cnvt_bool_to_uint8(img)
    # kernel = np.ones((3,3),np.uint8)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # dilation[dilation==255]=1

    return aspect, img, new_img


# feature 1
def crossing(img):
    cross = []
    # img = abs(1-img)
    # util.display_image(img)
    r, c = img.shape
    x = [int(r / 2)]
    y = [int(c / 4), int(3 * c / 4)]

    for i in x:
        cnt = 0
        sum = 0
        for j in range(0, c):
            if img[i, j] == 0:
                sum = sum + 1
            else:
                if (sum >= 1):
                    cnt = cnt + 1
                sum = 0
        cross.append(cnt)
    for j in y:
        cnt = 0
        sum = 0
        for i in range(0, r):
            if img[i, j] == 0:
                sum = sum + 1
            else:
                if (sum >= 1):
                    cnt = cnt + 1
                sum = 0
        cross.append(cnt)
    j = 0
    cnt = 0
    sum = 0
    for i in range(0, r):
        if img[i, j] == 0:
            sum = sum + 1
        else:
            if (sum >= 1):
                cnt += 1
            sum = 0
        j += 1
    cross.append(cnt)

    cnt = 0
    sum = 0
    j = c - 1
    for i in range(0, r):
        if img[i, j] == 0:
            sum = sum + 1
        else:
            if (sum >= 1):
                cnt = cnt + 1
            sum = 0
        j = j - 1
    cross.append(cnt)

    return cross


# feature 2
def histo(img):
    # img = abs(255 - img)
    img = cv2.resize(img, (25, 25), interpolation=cv2.INTER_CUBIC)
    img = pre.otsu_thresh(img)
    img[img == 255] = 1
    # util.display_image(img)
    vec = []
    vsum = np.sum(img, axis=0)
    hsum = np.sum(img, axis=1)
    for i in vsum:
        vec.append(i)
    for i in hsum:
        vec.append(i)
    return vec


# feature 3
def momentum(img):
    momentum_features = []
    for p in range(0, 4):
        for q in range(0, 4):
            x1, y1 = img.shape
            val = 0
            for x in range(0, x1):
                for y in range(0, y1):
                    if img[x][y] != 0:
                        val = val + ((x ** p) * (y ** q))
            momentum_features.append(val)

    moment = 0.0
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if (img[i][j] == 0):
                moment += math.cos(math.atan((i+1) * 1.0 / (j+1))) * math.sqrt((i+1) * (i+1) + (j+1) * (j+1))
    momentum_features.append(moment)
    return momentum_features


# feature 4
def calculate_pixels(img, x_start, x_end, y_start, y_end, val):
    cnt = 0
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            if img[x][y] != val:
                cnt = cnt + 1
    return cnt


def zoing_new(size, img):
    # create 5*5 zoning
    x, y = img.shape
    zone_features = []
    for i in range(0, size):
        for j in range(0, size):
            x1 = int(i * x / size)
            y1 = int(j * y / size)
            x2 = int(x1 + x / size)
            y2 = int(y1 + x / size)
            zone_features.append(calculate_pixels(img, x1, x2, y1, y2, 0))

    return zone_features


# feature 5
def cnt_hlines(img):
    r, c = img.shape
    l = []
    ans = 0
    ans1 = 0
    for i in range(0, r):
        cnt = 0
        for j in range(0, c):
            if img[i, j] == 1:
                cnt += 1
            else:
                if cnt >= 20:
                    ans += 1
                elif cnt >= 10:
                    ans1 += 1
                cnt = 0
    l.append(ans)
    l.append(ans1)
    return l


# feature 6
def cnt_vlines(img):
    r, c = img.shape
    ans = 0
    ans1 = 0
    l = []
    for j in range(0, c):
        cnt = 0
        for i in range(0, r):
            if img[i, j] == 1:
                cnt += 1
            else:
                if cnt >= 20:
                    ans += 1
                elif cnt >= 10:
                    ans1 += 1
                cnt = 0
    l.append(ans)
    l.append(ans1)
    return l


def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1

    l = []
    l.append(np.sum(out))
    # print(np.sum(out))
    return l, out
    # out[np.where(filtered==11)] = 1
    # return out


def count_endplt_regions(img):
    return zoing_new(3, img)


# counting loops in character

def find(i):
    if parent[i] != i:
        parent[i] = find(parent[i])
    return parent[i]


def union(i, j):
    parent[i] = j


def bounds(i, j, h, w):
    if (i >= 0 and i < h and j >= 0 and j < w):
        return 1
    return 0


def count_loops(img):
    global parent
    parent = []
    for i in range(0, 50 * 50):
        parent.append(i)
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
    img = pre.otsu_thresh(img)
    dir = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]]
    r, c = img.shape
    for i in range(0, r):
        for j in range(0, c):
            if img[i, j] == 0:
                continue
            for k in range(0, 8):
                x = i + dir[k][0]
                y = j + dir[k][1]
                if bounds(x, y, r, c) and img[x, y] != 0:
                    parenta = find(i * r + j)
                    parentb = find(x * r + y)
                    if parenta != parentb:
                        union(parenta, parentb)
    cnt = 0
    for i in range(0, r * c):
        if find(i) == i and img[int(i / r), i % r] != 0:
            cnt += 1
    l = []
    l.append(cnt)
    return l


##########################

# counting no of intersection points

def bounds2(i, j):
    if (i >= 0 and i < 3 and j >= 0 and j < 3):
        return 1
    return 0


def count_intersect(img):
    r, c = img.shape
    intersect = np.zeros((r, c), dtype=np.bool)
    for i in range(2, r - 2):
        for j in range(2, c - 2):
            x = i - 2
            y = j - 2
            neighbours = img[x:x + 5, y:y + 5]
            if np.sum(neighbours) > 7:
                intersect[i][j] = 1
    l = []
    l.append(np.sum(intersect))
    return l, intersect


###################################

def get_data(img):
    init_img = img.copy()
    aspect, old_img, img = pre_init(img)
    if aspect == -1:
        return -1
    img = util.cnvt_bool_to_uint8(img)
    features = []
    features.append(aspect)
    curr = histo(img)
    features += curr

    img[img == 255] = 1

    curr = crossing(img)
    features += curr
    curr = zoing_new(4, old_img)
    features += curr
    curr = momentum(img)
    features += curr
    curr = cnt_hlines(img)
    features += curr
    curr = cnt_vlines(img)
    features += curr
    curr, out = skeleton_endpoints(img)
    features += curr
    curr = count_endplt_regions(out)
    features += curr
    curr = count_loops(init_img)
    features += curr

    curr, intersect = count_intersect(img)
    features += curr
    curr = count_endplt_regions(intersect)
    features += curr

    return features
