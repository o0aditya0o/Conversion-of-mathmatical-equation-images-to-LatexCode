# -*- coding: utf-8 -*-

import cv2
import os
import preprocess as pre
import utility as util
import segmentation as seg
import features as fea
import pickle
import mapping as mp
import tolatex as tx
import alignmentchecker as ali
import utility as util

# input the image
def give_me_the_equation(img):
    import preprocess as pre
    # util.display_image(img)
    # print(path)
    # Converting light dark pixels ( <= 30 ) to black ( = 0 )
    img = pre.filter_image(img)
    # Thresholding OTSU

    img = pre.otsu_thresh(img)
    align = ali.align(img)
    # util.display_image(img)

    characters = seg.split_characters(img)
    print(len(characters));
    # for c in characters:
    #     util.display_image(c)

    with open('MLPClassifier.pkl', 'rb') as f:
        clf1 = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    ans = ""
    idx = 0
    pre = 0
    for c in characters:
        x = fea.get_data(c)
        temp = []
        temp.append(x)
        temp = scaler.transform(temp)
        y = clf1.predict(temp)
        y = mp.list[int(y[0])]
        if align[idx] == 0:
            if pre == 0:
                ans = ans + str(y)
            else:
                ans = ans + "}" + str(y)
        elif align[idx] == 1:
            if pre == 0:
                ans = ans + "^{" + str(y)
            else:
                ans = ans + str(y)
        else:
            if pre == 0:
                ans = ans + "_{" + str(y)
            else:
                ans = ans + str(y)
        print(y)
        pre = align[idx]
        idx = idx + 1
    if pre != 0:
        ans = ans + "}"
    print ("ADITYA")
    print(ans)
    return tx.tolatex(ans)





