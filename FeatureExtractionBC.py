import math
import numpy as np
from skimage.transform import radon
import cv2
import copy
import matplotlib.pyplot as plt

# Size parameters
height = 64
width = 64

# Filter Bank
FilterLevel = np.array([[1,4,6,4,1],
                        [4,16,24,16,4],
                        [6,24,32,24,6],
                        [4,16,24,16,4],
                        [1,4,6,4,1]],dtype=np.float32)

FilterEdge = np.array([[1,2,0,-2,-1],
                       [2,4,0,-4,-2],
                       [0,0,0,0,0],
                       [-2,-4,0,4,2],
                       [-1,-2,0,2,1]], dtype=np.float32)

FilterSpot = np.array([[1,0,-2,0,1],
                       [0,0,0,0,0],
                       [-2,0,-4,0,-2],
                       [0,0,0,0,0],
                       [1,0,-2,0,1]], dtype=np.float32)

FilterEdgeLevel1 = np.array([[-1,-2,0,2,1],
                             [-4,-8,0,8,4],
                             [-6,-12,0,12,6],
                             [-4,-8,0,8,4],
                             [-1,-2,0,2,1]], dtype=np.float32)

FilterEdgeLevel2 = np.array([[-1,-4,-6,-4,-1],
                             [-2,-8,-12,-8,-2],
                             [0,0,0,0,0],
                             [2,8,12,8,2],
                             [1,4,6,4,1]], dtype=np.float32)

FilterSpotLevel1 = np.array([[-1,0,2,0,-1],
                             [-4,0,8,0,-4],
                             [-6,0,12,0,-6],
                             [-4,0,8,0,-4],
                             [-1,0,2,0,-1]], dtype=np.float32)

FilterSpotLevel2 = np.array([[-1,-4,-6,-4,-1],
                             [0,0,0,0,0],
                             [2,8,12,8,2],
                             [0,0,0,0,0],
                             [-1,-4,-6,-4,-1]], dtype=np.float32)

FilterEdgeSpot1 = np.array([[1,0,-2,0,1],
                             [2,0,-4,0,2],
                             [0,0,0,0,0],
                             [-2,0,4,0,-2],
                             [-1,0,2,0,-1]], dtype=np.float32)

FilterEdgeSpot2 = np.array([[1,2,0,-2,-1],
                             [0,0,0,0,0],
                             [-2,-4,0,4,2],
                             [0,0,0,0,0],
                             [1,2,0,-2,-1]], dtype=np.float32)

# read images
currentImg = cv2.imread('2.png')
currentImg = cv2.resize(currentImg, (width,height))
lastImg = cv2.imread('1.png')
lastImg = cv2.resize(lastImg, (width,height))
# convert image to YCrCb and gray
currentGray = cv2.cvtColor(currentImg, cv2.COLOR_BGRA2GRAY)
lastGray = cv2.cvtColor(lastImg, cv2.COLOR_BGRA2GRAY)
currentImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2YCrCb)
lastImg = cv2.cvtColor(lastImg, cv2.COLOR_BGR2YCrCb)
# Split channel
currentY, currentCr, currentCb = cv2.split(currentImg)
lastY, lastCr, lastCb = cv2.split(lastImg)
# show channel
plt.imshow(currentY)
plt.show()
plt.imshow(currentCr)
plt.show()
plt.imshow(currentCb)
plt.show()

# compute Level on Y, Cb and Cr channel
LevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterLevel))/256
levelCr = np.abs(cv2.filter2D(currentCr, cv2.CV_32F, FilterLevel))/256
levelCb = np.abs(cv2.filter2D(currentCb, cv2.CV_32F, FilterLevel))/256
# compute Edge on Y channel
EgdeY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdge))/36
# compute Spot on Y channel
SpotY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterSpot))/16
# Compute Edge+Level on Y channel
EdgeLevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeLevel1))/192 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeLevel2))/192
# Compute Spot+Level on Y channel
SpotLevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterSpotLevel1))/128 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterSpotLevel2))/128
# Compute Spot+Edge on Y channel
SpotEdgeY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeSpot1))/48 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeSpot2))/48
# Compute Dense Optical flow
FlowY = cv2.calcOpticalFlowFarneback(prev=lastGray, next=currentGray,pyr_scale=0.5,levels=1,winsize=15,iterations=1,poly_n=5,poly_sigma=1.2,flags=0,flow=None)
print("plop")
# Plot feature
plt.imshow(LevelY)
plt.show()
plt.imshow(levelCr)
plt.show()
plt.imshow(levelCb)
plt.show()
plt.imshow(EgdeY)
plt.show()
plt.imshow(SpotY)
plt.show()
plt.imshow(EdgeLevelY)
plt.show()
plt.imshow(SpotLevelY)
plt.show()
plt.imshow(SpotEdgeY)
plt.show()
plt.imshow(FlowY[:,:,0])
plt.show()
plt.imshow(FlowY[:,:,1])
plt.show()
# 8x8x8=512 Spatial GIST feature (average or median)

# R-KNN classifier ?
