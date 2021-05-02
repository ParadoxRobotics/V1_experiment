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
    dx = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img,(5,5),0), cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img,(5,5),0), cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1)
    """
    # compute second order derivative in x and y
    dxxFilter = np.array([[1],[-2],[1]])
    dyyFilter = np.array([[1,-2,1]])
    dxx = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),-1, dxxFilter)
    dyy = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),-1, dyyFilter)
    """
    # avoid division by zero
    dx = dx+0.000001
    dy = dy+0.000001
    #dxx = dxx+0.000001
    #dyy = dyy+0.000001
    # compute magnitude of the edge
    magnitude = cv2.magnitude(dx, dy)
    # compute magnitude mask to remove noise
    ret, magMask = cv2.threshold(magnitude, th, 255, cv2.THRESH_BINARY)
    magMask = magMask.astype(np.uint8)
    # apply the mask to the magnitude (divide by the max soble response -> normalize to [0,1])
    magnitude = cv2.bitwise_and(magnitude, magnitude, mask=magMask) / 1448
    # compute the angle given the thresholding magnitude mask [0, 360]
    angle = cv2.phase(dx, dy, angleInDegrees=True)
    angle = cv2.bitwise_and(angle, angle, mask=magMask)
    # compute contour curvature given the magnitude mask
    contour, H = cv2.findContours(magMask.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
    # init curvature image
    curvature = np.zeros_like(magnitude)
    # for every contour
    for c in contour:
        contourSize = c.shape[0]
        if contourSize > 10:
            for i in range(c.shape[0]):
                # get point from the contour
                x1 = c[i][0][0]
                y1 = c[i][0][1]
                x2 = c[(i+nbPointCurv)%contourSize][0][0]
                y2 = c[(i+nbPointCurv)%contourSize][0][1]
                # get gradient
                gx1 = dx[y1, x1]
                gy1 = dy[y1, x1]
                gx2 = dx[y2, x2]
                gy2 = dy[y2, x2]
                # compute angle between gradient
                cos_angle = gx1 * gx2 + gy1 * gy2
                cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
                varAngle = np.arccos(cos_angle)
                if cos_angle < 0:
                    varAngle = np.pi - varAngle
                # get the middle point
                x1 = c[((2*i+nbPointCurv)//2)%contourSize][0][0]
                y1 = c[((2*i+nbPointCurv)//2)%contourSize][0][1]
                # update the curvature heatmap
                if varAngle < 0 or np.isnan(varAngle):
                    varAngle = 0.0
                curvature[y1, x1] = varAngle
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

# extract integral feature
def integral_feature(img, lastIntegral, tau):
    return  tau*img + (1-tau)*lastIntegral


# compute kronecker function kernel
def kronecker(v,c):
    if v == c:
        return 1
    else:
        return 0

# Quadratic B-spline function kernel
def quad_B_spline(v):
    if np.abs(v) <= 0.5:
        return (3/4)-(v**2)
    if np.abs(v) > 0.5 and np.abs(v) < (3/2):
        return ((v-(3/2))**2)/2
    if np.abs(v) >= (3/2):
        return 0

# linear B-spline function kernel
def linear_B_spline(v):
    if np.abs(v) >= 1:
        return 1
    else:
        return 0

# squared cosine function kernel
def squared_cosine(v):
    if np.abs(v) <= 3/2:
        return (2/3)*((np.cos(np.pi*v/3))**2)
    else:
        return 0

# rectangular function kernel
def rectangular(v):
    if np.abs(v) < 0.5:
        return 1
    elif np.abs(v) == 0.5:
        return 0.5
    else:
        return 0

# Size parameters
height = 128
width = 128

# read images
current_img = cv2.imread('cur.png')
current_img = cv2.resize(current_img, (width,height))
last_img = cv2.imread('ref.png')
last_img = cv2.resize(last_img, (width,height))

# compute feature
hue, saturation, intensity = color_feature(img=current_img, blurrLevel=3)
magnitude, angle, curvature = edge_feature(img=current_img, th=80, nbPointCurv=8)
variability, velocity, direction = flow_feature(refImg=last_img, curImg=current_img, th=20)

plt.imshow(intensity)
plt.show()

kfMap = np.zeros(intensity.shape)
bsMap = np.zeros(intensity.shape)
lbMap = np.zeros(intensity.shape)
scMap = np.zeros(intensity.shape)
reMap = np.zeros(intensity.shape)
n=250
for h in range(0, angle.shape[0]):
    for w in range(0, angle.shape[1]):
        kfMap[h,w] = kronecker(intensity[h,w], n)
        bsMap[h,w] = quad_B_spline(intensity[h,w]-n)
        lbMap[h,w] = linear_B_spline(intensity[h,w]-n)
        scMap[h,w] = squared_cosine(intensity[h,w]-n+3/2)
        reMap[h,w] = rectangular(intensity[h,w]-n)

plt.imshow(kfMap)
plt.show()
plt.imshow(bsMap)
plt.show()
plt.imshow(lbMap)
plt.show()
plt.imshow(scMap)
plt.show()
plt.imshow(reMap)
plt.show()

"""
# plot main feature
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
