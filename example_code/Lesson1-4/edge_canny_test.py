#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 10:26:03 2018

@author: raffa
"""

import cv2
import numpy as np

# Read in the image
#image = mpimg.imread('test.jpg')
gray = cv2.imread('test.jpg',0)

low_threshold = 100
high_threshold = 200

edge = cv2.Canny(gray, low_threshold, high_threshold)

# take the negative image
edge = np.abs(edge - 255*np.ones(edge.shape))


cv2.imshow('edge',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()