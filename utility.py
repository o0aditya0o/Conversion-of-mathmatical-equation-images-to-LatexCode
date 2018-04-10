import cv2
import numpy as np


def display_image(img, str='image'):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cnvt_bool_to_uint8(img):
    ret = np.zeros(img.shape, dtype=np.uint8)
    r, c = img.shape
    for i in range(0,r):
        for j in range(0,c):
            if img[i,j] == 1:
                ret[i,j] = 255
    return ret
