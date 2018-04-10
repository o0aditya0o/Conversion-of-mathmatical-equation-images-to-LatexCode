# -*- coding: utf-8 -*-

import cv2
import os
from features import get_data
from utility import display_image

x = 0
y = 0
cwd = os.getcwd()
train_in = '4train_input.txt'
test_in = '4test_input.txt'
train_out = '4train_out.txt'
test_out = '4test_out.txt'
train_in = open(train_in, 'w')
train_out = open(train_out, 'w')
test_in = open(test_in, 'w')
test_out = open(test_out, 'w')

d = dict()

symbols = ['+', '-', '/', '*', '(', ')', '[', ']', '{', '}', 'α', 'β', 'γ', 'θ', 'η', 'μ', 'λ', 'π', 'ρ', 'σ', 'τ', 'δ',
           'ϕ', '≤', '≥', '≠', '÷', '×', '±', '∑', '∫', '∏']
pos = 1

path = cwd+'/dataset/'
for files in os.listdir(path):
    real_value = pos
    print(files)
    print(real_value)
    if pos < 95:
        pos = pos + 1
        continue
    path2 = os.listdir(path  + files+'/')
    y = (len(path2))
    cnt = 0
    boudary = (y * 90) / 100
    for imges in path2:
        val = path+files+'/' + imges
        img = cv2.imread(val, 0)
        cnt = cnt + 1
        new_list = get_data(img)
        if new_list == -1:
            continue
        # print (new_list)
        if cnt > boudary:
            for i in new_list:
                test_in.write(str(i))
                test_in.write(" ")
            test_in.write('\n')
            test_out.write(str(real_value))
            test_out.write('\n')
        else:
            for i in new_list:
                train_in.write(str(i))
                train_in.write(" ")
            train_in.write('\n')
            train_out.write(str(real_value))
            train_out.write('\n')
    pos = pos + 1

train_out.close()
train_in.close()
test_out.close()
test_in.close()
