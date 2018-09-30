# -*- coding: utf-8 -*-

'''
Once you have defined your upper and lower limits, 
you then make a call to the cv2.inRange method which returns a mask, 
specifying which pixels fall into your specified upper and lower range.
The mask is defined by two different numer 0 and 255.
0 means the corresponding element in the image is not in the range 
while 255 means it's in.

Finally, now that you have the mask, 
you can apply it to your image using the cv2.bitwise_and function.
In the function you are using the same image because that way bitwise_and
function does not change the image itself

It is possible to work even on colored image using lower and upper bounds as
3 elements arrays each on BGR color space.

See https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
for more info.
'''

import cv2
import numpy as np

test_gray_img_01 = np.random.randint(0,255,100).reshape(10,10)
test_gray_img_02 = np.random.randint(0,255,100).reshape(10,10)


print('test_gray_img_01\n', test_gray_img_01)

print('test_gray_img_02\n', test_gray_img_02)

lower = np.array([51], np.uint8)
upper = np.array([137], dtype='uint8')

mask_01 = cv2.inRange(test_gray_img_01,lower, upper)

print('mask01\n', mask_01)

mask_02 = cv2.inRange(test_gray_img_02,lower,upper)

print('mask02\n', mask_02)

output_01 = cv2.bitwise_and(test_gray_img_01,test_gray_img_01,mask=mask_01)

print('final output_01\n', output_01)

output_02 = cv2.bitwise_and(test_gray_img_02,test_gray_img_02,mask=mask_02)

print('final output_02\n', output_02)