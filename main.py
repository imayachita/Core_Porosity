import json
import pandas as pd
import numpy as np

import os
from tqdm import tqdm
import argparse
import re
import cv2
import collections

from tqdm import tqdm

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", type=str, required=False, help="Image")
    ap.add_argument("-o", type=str, required=False, help="output dir")

    args = ap.parse_args()

    im = args.i
    output = args.o

    im = cv2.imread(im)
    result = im.copy()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    #HSV filtering
    lower_black = np.array([0,0,0])
    upper_black = np.array([105,105,105])

    #create mask with HSV filtering
    mask_hsv = cv2.inRange(im, lower_black, upper_black)
    res = cv2.bitwise_and(im,im,mask=mask_hsv)

    #get the core image from the whole image
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    h,w,_ = im.shape
    mask = np.zeros((h+2,w+2), np.uint8)

    _,binary = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
            cnt2 = np.vstack(cnt).squeeze()
            _,binary,_,_ = cv2.floodFill(binary, mask, tuple(cnt2[0]), 0)

    # convert image back to original color
    binary = cv2.bitwise_not(binary)

    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.0001*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    core_area = cv2.contourArea(approx)
    w,h = mask.shape

    tot_area = w*h
    bkg = tot_area - core_area

    print('core area: ', core_area)
    print('total image area: ', tot_area)
    print('background area: ', bkg)

    # cv2.drawContours(im, cnt, -1, (0,255,0), 2)
    cv2.drawContours(im, [approx], -1, (0, 0, 255), 2)

    # tot_white_pix = (mask_hsv==1).sum()
    tot_black_pix = (mask_hsv==0).sum()

    # matrix_percent = tot_black_pix/core_area
    # porosity = 1-matrix_percent
    #
    # print('total white: ', tot_white_pix)
    # print('total black: ', tot_black_pix)
    # print('Porosity: ', porosity)

    #find the circle that encloses the contour to find core area
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    im_copy = im.copy()
    img_crc = cv2.circle(im_copy,center,radius,(0,200,20),2)
    circle_area = 3.14*radius**2
    print('circle area: ', circle_area)
    print('Porosity: ', 1-(tot_black_pix/circle_area))



    cv2.imshow('gray', imgray)
    cv2.imshow('image', im)
    cv2.imshow('mask', mask_hsv)
    cv2.imshow('circle',img_crc)
    cv2.imwrite('mask_hsv.png',mask_hsv)

    k = cv2.waitKey()
    if k == 27:
        cv2.destroyAllWindows()
