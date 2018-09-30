#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:59:49 2018
"""

# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Read in and grayscale the image
image = mpimg.imread('imgs/exit-ramp.jpg')
# image = mpimg.imread('imgs/solidWhiteRight.jpg')
# image = mpimg.imread('imgs/solidWhiteCurve.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
print(gray.shape)

#%%
# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

#%%
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#%%
# Define the Hough transform parameters
# Make a blank image of the same size as our image to draw on
rho = 5
theta = np.pi/180
threshold = 100
min_line_length = 10
max_line_gap = 10
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

#%%
# Iterate over the output "lines" and draw lines on the blank image
line_color = [255, 0, 0]
line_thickness = 5

pos_lines_list = []
neg_lines_list = []

# Create a sensible polygonal area
upper_limit_pixel = 330
bottom_left_vert = [70,image.shape[0]]
upper_left_vert = [460,upper_limit_pixel]
upper_right_vert = [540,upper_limit_pixel]
bottom_right_vert = [900,image.shape[0]]
sens_poly_vertices = np.array([bottom_left_vert , upper_left_vert , upper_right_vert , bottom_right_vert ], np.int32)
sens_polygon = Polygon(sens_poly_vertices)

for line in lines:
    for x1,y1,x2,y2 in line:
        # Select only the lines whose points are within the area of the polygon
        test_point1 = Point(x1,y1)
        test_point2 = Point(x2,y2)
        if sens_polygon.contains(test_point1) and sens_polygon.contains(test_point2):
            i_gradient = (y2 - y1) / (x2 - x1)        
            i_intercept = (x2*y1 - x1*y2) / (x2 - x1)            
            print('y = {:+.2f}*x + {:+.2f}'.format(i_gradient,i_intercept))
            if i_gradient > 0.1 and i_gradient < 1:
                pos_lines_list.append((i_gradient,i_intercept))
            if i_gradient < -0.1 and i_gradient > -1:
                neg_lines_list.append((i_gradient,i_intercept))
            
lines_data = []
if len(pos_lines_list) != 0:
    lines_data.append(np.median(pos_lines_list,axis=0))
if len(neg_lines_list) != 0:
    lines_data.append(np.median(neg_lines_list,axis=0))

for i_line_data in lines_data:
    print(i_line_data)    
    # Draw line
    y1 = image.shape[0]
    x1 = int((y1 - i_line_data[1]) / i_line_data[0])
    y2 = upper_limit_pixel
    x2 = int((y2 - i_line_data[1]) / i_line_data[0])
    cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_thickness)


#%%
# line_image is a (540, 960, 3) "color" image, wiht the same size of original image.
# In order to fuse line_image and masked_edges (the output of Canny operation),
# it is necessary first to create a fake "color" binary image by stacking masked_edges 3 times,
# one for each color in BGR space.
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 


# Draw the lines on the edge image by fusing color_edges and line_image.
# cv2.addWeighted execute this operation st = src1*alpha + src2*beta + gamma 
# on each element of the images
combo = cv2.addWeighted(color_edges, 0.5, line_image, 1, 0) 

plt.imshow(combo)