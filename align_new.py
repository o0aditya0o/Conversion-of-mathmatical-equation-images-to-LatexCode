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

def align(img, model)