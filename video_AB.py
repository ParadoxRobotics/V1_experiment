import math
import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt

# extract color features -> Hue, saturation and intensity
def color_feature(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, saturation, intensity = cv2.split(hsvImg)
    return hue, saturation, intensity

# extract edge feature -> magnitude, angle and curvature
def edge_feature(img, th, nbPointCurv):
    # compute derivative in x and y
    dx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
    # avoid division by zero
    dx = dx+0.000001
    dy = dy+0.000001
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
    contour, H = cv2.findContours(magMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # init curvature image
    curvature = np.zeros_like(magnitude)
    # for every contour
    for c in contour:
        contourSize = c.shape[0]
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
            curvature[y1, x1] = varAngle

    return magnitude, angle, curvature

# compute flow feature -> variability, velocity and drection
# compute flow feature -> variability and velocity
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

# init camera
cam = WebcamVideoStream(src=0).start()
refFrame = cam.read()
refFrame = cv2.resize(refFrame, (320,240))

# compute position feature
xLoc = np.ones((refFrame.shape[0], refFrame.shape[1]))
yLoc = np.ones((refFrame.shape[0], refFrame.shape[1]))
xLoc[:,:] = xLoc[:,:]*np.arange(refFrame.shape[1])
yLoc[:,:] = yLoc[:,:]*np.arange(refFrame.shape[0]).reshape((-1,1))

while True:
    # get current frame
    curFrame = cam.read()
    curFrame = cv2.resize(curFrame, (320,240))
    # compute feature
    hue, saturation, intensity = color_feature(img=curFrame)
    magnitude, angle, curvature = edge_feature(img=curFrame, th=80, nbPointCurv=2)
    variability, velocity, direction = flow_feature(refImg=refFrame, curImg=curFrame, th=50)
    # plot process
    feature_color_concat = np.concatenate((np.uint8(hue), np.uint8(saturation), np.uint8(intensity)), axis=1)
    feature_edge_concat = np.concatenate((np.uint8(cv2.normalize(magnitude, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(angle, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(curvature, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))), axis=1)
    feature_flow_concat = np.concatenate((np.uint8(cv2.normalize(variability, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(velocity, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(direction, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))), axis=1)
    feature_position_concat = np.concatenate((np.uint8(cv2.normalize(xLoc, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(yLoc, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)),
                                          np.uint8(cv2.normalize(np.zeros((refFrame.shape[0], refFrame.shape[1])), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))), axis=1)
    feature_concat = np.concatenate((feature_color_concat, feature_edge_concat, feature_flow_concat, feature_position_concat), axis=0)

    cv2.imshow('feature', feature_concat)
    # update image
    refFrame = curFrame

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
