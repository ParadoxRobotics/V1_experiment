import math
import numpy as np
import cv2
import imutils
import copy
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt

# extract color features -> Hue, saturation and intensity
def color_feature(img, blurrLevel):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, saturation, intensity = cv2.split(hsvImg)
    return cv2.medianBlur(hue, blurrLevel), cv2.medianBlur(saturation, blurrLevel), intensity

# extract edge feature -> magnitude, angle and curvature
def edge_feature(img, th, nbPointCurv):
    # compute first order derivative in x and y
    dx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
    # compute second order derivative in x and y
    dxxFilter = np.array([[1],[-2],[1]])
    dyyFilter = np.array([[1,-2,1]])
    dxx = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),-1, dxxFilter)
    dyy = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),-1, dyyFilter)
    # avoid division by zero
    dx = dx+0.000001
    dy = dy+0.000001
    dxx = dxx+0.000001
    dyy = dyy+0.000001
    # compute magnitude of the edge
    magnitude = cv2.magnitude(dx, dy)
    # compute magnitude mask to remove noise
    ret, magMask = cv2.threshold(magnitude, th, 255, cv2.THRESH_BINARY)
    magMask = magMask.astype(np.uint8)
    # apply the mask to the magnitude
    magnitude = cv2.bitwise_and(magnitude, magnitude, mask=magMask)
    # compute the angle given the thresholding magnitude mask
    angle = cv2.phase(dx, dy, angleInDegrees=True)
    angle = cv2.bitwise_and(angle, angle, mask=magMask)
    # compute contour curvature given the magnitude mask
    return magnitude, angle, curvature

# compute flow feature -> variability, velocity and direction
def flow_feature(refImg, curImg, th):
    # compute difference
    diff = cv2.subtract(cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY), cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY))
    ret, diffMask = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)
    diffMask = diffMask.astype(np.uint8)
    diff = cv2.bitwise_and(diff, diff, mask=diffMask)
    # compute sum
    add = cv2.add(cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY), cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY))
    # create filter
    dg =  np.array(([1,0,-1]), dtype="float32")
    gd =  np.array(([0.2163,0.5674,0.2163]), dtype="float32") # norm = 1
    # compute component
    dx = cv2.sepFilter2D(add, cv2.CV_32F, gd, dg)
    dy = cv2.sepFilter2D(add, cv2.CV_32F, dg, gd)
    # compute mag ang direction
    mag, angle = cv2.cartToPolar(dy, dx)
    # compute mag mask
    velocity = cv2.bitwise_and(mag, mag, mask=diffMask)
    direction = cv2.bitwise_and(angle, angle, mask=diffMask)*180/np.pi/2
    return diff, velocity, direction

"""
# compute kronecker bin function
def kronecker(img, bin):
    # init output
    dm = np.zeros(img.shape)
    # perform kronecker operation
    for h in range(0, img.shape[0]):
        for w in range(0, img.shape[1]):
            if img[h,w] == bin.any():
"""

# Size parameters
height = 480
width = 640

# read images
current_img = cv2.imread('car2.jpg')
current_img = cv2.resize(current_img, (width,height))
last_img = cv2.imread('car1.jpg')
last_img = cv2.resize(last_img, (width,height))

# compute feature
hue, saturation, intensity = color_feature(img=current_img, blurrLevel=3)
magnitude, angle, curvature = edge_feature(img=current_img, th=80, nbPointCurv=5)
variability, velocity, direction = flow_feature(refImg=last_img, curImg=current_img, th=20)

"""
plt.imshow(hue)
plt.show()
plt.imshow(saturation)
plt.show()
plt.imshow(intensity)
plt.show()
plt.imshow(magnitude)
plt.show()
plt.imshow(angle)
plt.show()
plt.imshow(curvature)
plt.show()
plt.imshow(variability)
plt.show()
plt.imshow(velocity)
plt.show()
plt.imshow(direction)
plt.show()
"""
