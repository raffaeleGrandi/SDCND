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

# Read in and grayscale the image
# image = mpimg.imread('imgs/exit-ramp.jpg')
# image = mpimg.imread('imgs/solidWhiteRight.jpg')
image = mpimg.imread('imgs/solidWhiteCurve.jpg')
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
rho = 3
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
line_color=[255, 0, 0]
line_thickness=5


for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_thickness)
          
#%%

# Create a sensible polygonal area
upper_limit_pixel = 330
bottom_left_vert = [70,image.shape[0]]
upper_left_vert = [460,upper_limit_pixel]
upper_right_vert = [540,upper_limit_pixel]
bottom_right_vert = [900,image.shape[0]]
sens_poly_vertices = np.array([bottom_left_vert , upper_left_vert , 
                               upper_right_vert , bottom_right_vert ], np.int32)

sens_poly_color = [255,255,255]

# Create a blank image on which draw the polygon
blank_img = np.copy(image)*0
sens_image = cv2.fillPoly(blank_img,[sens_poly_vertices], sens_poly_color,1)

# Now we want detect where the color is below sens_poly_color, creating a True matrix. 
# Having drawn the sensible polygon on a blank image, the matrix elements are True
# only in correspondance of the region NOT in the polygon (neg_sens_threshold_mask)
neg_sens_threshold_mask = (sens_image[:,:,0] < sens_poly_color[0]) \
                        | (sens_image[:,:,1] < sens_poly_color[1]) \
                        | (sens_image[:,:,2] < sens_poly_color[2])

# line_image is a blank image on which we have drawn the lines coming from the Hough
# transformation.
# We want to select the area of line_image corresponding to the 
# sensible polygon, applying the True mask given by neg_sens_threshold_mask,
# it is possible to put to zero the elements outside the sensible area.
line_image[neg_sens_threshold_mask] = [0,0,0]

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