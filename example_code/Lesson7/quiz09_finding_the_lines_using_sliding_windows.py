#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:45:32 2018

@author: raffa
"""

'''
##### Lesson7
'''

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
binary_warped = mpimg.imread('imgs/warped-example.jpg')/255

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    half_index = img.shape[0]//2
    bottom_half = img[half_index:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half,axis=0)
    
    return histogram


def find_lane_pixels(binary_warped):
    # Create histogram of image binary activations
    histogram = hist(binary_warped)
    
    # Visualize the resulting histogram
    plt.plot(histogram)
    
    # -------
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    print('Left lane is around row {}'.format(leftx_base))
    print('Right lane is around row {}'.format(rightx_base))
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    '''
    Now that we've set up what the windows look like and have a starting point, 
    we'll want to loop for nwindows, with the given window sliding left or right 
    if it finds the mean position of activated pixels within the window to have 
    shifted.
    '''
    
    # Step through the windows one by one
    for window in range(nwindows):
        # print('window number {}'.format(window))
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height # pixel che identifica il bordo alto
        win_y_high = binary_warped.shape[0] - window*window_height # pixel che identifica il bordo basso
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 3) 
       
    
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        true_map_xleft = (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        true_map_xright = (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        true_map_y = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
        
        true_map_leftwin = true_map_xleft & true_map_y
        true_map_rightwin = true_map_xright & true_map_y    
        
        good_left_inds = np.where(true_map_leftwin)[0]        
        # good_left_inds_solu = true_map_leftwin.nonzero()[0] # code from solution
        # print(min(good_left_inds_solu == good_left_inds))
        #print('number of good left: {}'.format(good_left_inds.shape))
        good_right_inds = np.where(true_map_rightwin)[0]
        # print('number of good right: {}'.format(good_right_inds.shape))
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:        
            # prima recupero il valore delle colonne dove i pixel sono nonzero 
            # sulla finestra di sinistra
            good_left_cols = nonzerox[good_left_inds]
            # poi uso il valore delle colonne dei pixel per ricentrare 
            # la finestra di sinistra
            leftx_current = np.int(np.mean(good_left_cols))
        if len(good_right_inds) > minpix:        
            good_right_cols = nonzerox[good_right_inds]
            rightx_current = np.int(np.mean(good_right_cols))
    
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)