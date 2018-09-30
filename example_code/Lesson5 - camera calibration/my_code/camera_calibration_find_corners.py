import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 9#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
# fname = 'chessboard_imgs/2018-08-13-112448.jpg'
fname = 'imgs/chessboard_imgs/test_image.jpg'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
print('Corners found: ',ret)
# If found, draw corners
if ret == True:
    # Draw and display the corners
    index = [0,8,45,53]
    print(corners[index])
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
