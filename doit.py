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
def give_me_the_equation(img, model):
    if len(img) == 0 or len(img[0]) == 0:
        return ""
    import preprocess as pre
    util.display_image(img, 'do_ittttt')
    # print(path)
    # Converting light dark pixels ( <= 30 ) to black ( = 0 )
    img = pre.filter_image(img)
    # Thresholding OTSU

    img = pre.otsu_thresh(img)
    align = ali.align(img, model)
    print(align)
    # util.display_image(img)

    characters = seg.split_characters(img)
    # for c in characters:
    #     util.display_image(c)

    ans = ""
    idx = 0
    pre = 0
    for c in characters:
        y = util.predict_class(c, model)
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
        # print(y)
        pre = align[idx]
        idx = idx + 1
    if pre != 0:
        ans = ans + "}"
    # print ("ADITYA")
    # print(ans)
    return tx.tolatex(ans)





