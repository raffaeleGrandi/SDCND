#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:30:35 2018

@author: raffa
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_data():  

    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720) # to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    
    # Viene utilizzato un polinomio del secondo ordine per generare i punti 'fake'
    # appartenenti alle lane lines, inoltre
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    x_range = 50
    left_lane_base_x = 200
    leftx = np.array([left_lane_base_x + (y**2)*quadratic_coeff + 
                      np.random.randint(-x_range, high=x_range+1) for y in ploty])
    
    right_lane_base_x = 900
    rightx = np.array([right_lane_base_x + (y**2)*quadratic_coeff + 
                       np.random.randint(-x_range, high=x_range+1) for y in ploty])
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y    
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)    
    right_fit = np.polyfit(ploty, rightx, 2)    
    
    return ploty, left_fit, right_fit, leftx, rightx


# ------

def generate_lane():    
     # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)
        
    ''' Part 1: generating fake line lanes data '''
    
    ploty, left_fit, right_fit, leftx, rightx = generate_data()
    
    # Generate x values for the fitted second order polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]    
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.figure(figsize=(1280, 720))
    #plt.savefig('test_plot.jpg', dpi=10)
    return ploty, left_fitx, right_fitx


#-----

ploty, left_fitx, right_fitx = generate_lane()
plt.close()

warped = cv2.imread('binary_warp2.jpg')[:,:,0]
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
print(color_warp.shape)

#plt.imshow(color_warp)

#ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

lane_color = (0,255,0)

#px = [70,430,540,900]
#py = [700,330,330,700]

px = np.concatenate([left_fitx,np.flipud(right_fitx)])
py = np.concatenate([ploty,np.flipud(ploty)])

pts = np.array( [[x,y] for x,y in zip(px,py)], np.int32)
cv2.fillPoly(color_warp, [pts], lane_color)

plt.imshow(color_warp)

'''
cv2.fillPoly(mask, vertices, ignore_mask_color)

upper_limit_pixel = 330
    bottom_left_vert = [70,initial_img.shape[0]]
    upper_left_vert = [430,upper_limit_pixel]
    upper_right_vert = [540,upper_limit_pixel]
    bottom_right_vert = [900,initial_img.shape[0]]
    sens_poly_vertices = np.array([bottom_left_vert , upper_left_vert , upper_right_vert , bottom_right_vert ], np.int32)

    masked_image = region_of_interest(img_lines, [sens_poly_vertices])
'''
