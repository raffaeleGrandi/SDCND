import numpy as np
import cv2
import glob
import pickle

nx = 7 # num corners on chessboard horizontal axis
ny = 6 # num corners on chessboard vertical axis

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(nx,ny,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('imgs/chessboard_imgs/*.jpg')
print('Found {} images\n'.format(len(images)))
img_counter = 0

for fname in images:
    img_counter += 1
    print('Processing image {}'.format(img_counter))
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)


    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx,ny), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

print('Saving points data...')
wide_dist_pickle = {}
wide_dist_pickle['objpoints'] = objpoints
wide_dist_pickle['imgpoints'] = imgpoints

pickle_out = open('wide_dist_pickle.pkl','wb')
pickle.dump(wide_dist_pickle, pickle_out)
pickle_out.close()