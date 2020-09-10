import numpy as np
import cv2
import matplotlib.pyplot as plt

# compute flow feature -> variability and velocity
def flow_feature(refImg, curImg, tau, th):
    variability = cv2.absdiff(cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY), cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY))
    plt.imshow(variability)
    plt.show()
    xv = np.array((
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]), dtype="int")
    yv = np.array((
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]), dtype="int")
    dx = cv2.filter2D(variability, -1, xv)
    dy = cv2.filter2D(variability, -1, yv)
    # compute magnitude
    mag = cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32))
    ret, magMask = cv2.threshold(mag, th, 255, cv2.THRESH_BINARY)
    magMask = magMask.astype(np.uint8)
    mag = cv2.bitwise_and(mag, mag, mask=magMask)
    # compute angle
    angle = cv2.phase(dx.astype(np.float32), dy.astype(np.float32))
    # remove noise using a binary mask
    ret, varMask = cv2.threshold(variability, th, 255, cv2.THRESH_BINARY)
    varMask = varMask.astype(np.uint8)
    variability = cv2.bitwise_and(variability, variability, mask=varMask)
    variability = cv2.normalize(variability,  None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return variability, dx, dy, mag, angle

# compute flow feature -> variability and velocity
def flow_farnback(refImg, curImg, th):
    variability = cv2.absdiff(cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY), cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY))
    # calculate dense optical flow with the last and current image
    OF = cv2.calcOpticalFlowFarneback(prev=cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY),
                                      next=cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY),
                                      pyr_scale=0.5,
                                      levels=3,
                                      winsize=15,
                                      iterations=3,
                                      poly_n=5,
                                      poly_sigma=1.2,
                                      flags=0,flow=None)
    fx = OF[:,:,0]
    fy = OF[:,:,1]

    mag = cv2.magnitude(fx, fy)

    angle = cv2.phase(fx, fy)

    ret, varMask = cv2.threshold(variability, th, 255, cv2.THRESH_BINARY)
    varMask = varMask.astype(np.uint8)
    # Mask farnback
    mag = cv2.bitwise_and(mag, mag, mask=varMask)
    angle = cv2.bitwise_and(angle, angle, mask=varMask)

    return variability, fx, fy, mag, angle

refImg = cv2.imread('ref.png')
refImg = cv2.resize(refImg, (640, 480))
curImg = cv2.imread('cur.png')
curImg = cv2.resize(curImg, (640, 480))

var, fx, fy, mag, angle = flow_farnback(refImg=refImg, curImg=curImg, th=10)

plt.imshow(var)
plt.show()
plt.imshow(fx)
plt.show()
plt.imshow(fy)
plt.show()
plt.imshow(mag)
plt.show()
plt.imshow(angle)
plt.show()
