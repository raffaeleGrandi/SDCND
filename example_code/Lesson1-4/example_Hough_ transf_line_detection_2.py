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

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from sklearn.cluster import KMeans

# Read in and grayscale the image
# image = mpimg.imread('imgs/exit-ramp.jpg')
# image = mpimg.imread('imgs/solidWhiteRight.jpg')
image = mpimg.imread('imgs/solidYellowCurve.jpg')
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
rho = 1
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
lines_data = []

# Create a sensible polygonal area
bottom_left_vert = [70,540]
upper_left_vert = [460,300]
upper_right_vert = [490,300]
bottom_right_vert = [900,540]
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
            lines_data.append((i_gradient,i_intercept))
            print('y = {:+.2f}*x + {:+.2f}'.format(i_gradient,i_intercept))
            if i_gradient > 0:
                pos_lines_list.append((i_gradient,i_intercept))
            if i_gradient < 0:
                neg_lines_list.append((i_gradient,i_intercept))
            

if len(pos_lines_list) != 0:
    pos_line_data = np.mean(pos_lines_list,axis=0) 
    print(pos_line_data)
    # Draw positive line
    y1 = 540
    x1 = int((y1 - pos_line_data[1]) / pos_line_data[0])
    y2 = 300
    x2 = int((y2 - pos_line_data[1]) / pos_line_data[0])
    cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_thickness)

if len(neg_lines_list) != 0:
    neg_line_data = np.mean(neg_lines_list,axis=0)
    print(neg_line_data)
    # Draw negative line
    y1 = 540
    x1 = int((y1 - neg_line_data[1]) / neg_line_data[0])
    y2 = 300
    x2 = int((y2 - neg_line_data[1]) / neg_line_data[0])
    cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_thickness)

kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(lines_data))
print(kmeans.cluster_centers_)

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