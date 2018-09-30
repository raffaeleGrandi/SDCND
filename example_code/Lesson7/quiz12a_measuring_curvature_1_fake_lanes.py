#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:06:52 2018

@author: raffa
"""

import numpy as np
import matplotlib.pyplot as plt

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


def measure_curvature_pixels(ploty, left_fit, right_fit):
    
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!    
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    ## Implement the calculation of the left line here
    left_curverad = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2)) / np.abs(2*left_fit[0])
    ## Implement the calculation of the right line here
    right_curverad = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2)) / np.abs(2*right_fit[0])
    
    # Solution
    #left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    #right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    
    return left_curverad, right_curverad

# ------
    
 # Set random seed number so results are consistent for grader
# Comment this out if you'd like to see results on different random data!
np.random.seed(0)
    
''' Part 1: generating fake line lanes data '''

ploty, left_fit, right_fit, leftx, rightx = generate_data()

# Generate x values for the fitted second order polynomials
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]    
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

test_fit = np.copy(right_fit)+[0.1e-04, 2e-01, 3]
test_fitx = test_fit[0]*ploty**2 + test_fit[1]*ploty + test_fit[2]    
# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.plot(test_fitx, ploty, color='yellow', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images

''' Part2: measuring curvature '''

left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
print(left_curverad, right_curverad)

print(left_fit)
print(right_fit)

print(np.mean(rightx - leftx, axis=0))
    
