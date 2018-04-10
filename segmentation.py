from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import numpy as np

def split_characters(img):
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

    # plotting minimas in histogram i.e. word breaks
    # for i in minn:
    #     plt.plot(i, y_smooth[i], 'ro')
    # plt.show()

    # creating a list of chars
    clist = []
    prev = -1
    for i in minn:
        if (prev != -1):
            ch = img[:, prev:i + 1]
            ch_not = abs(255 - ch)
            if (np.sum(ch_not) >= 6 * 255):
                clist.append(ch)
        prev = i

    return clist
