import math
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

# Size parameters
height = 512
width = 512

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
currentImg = cv2.imread('output.png')
currentImg = cv2.resize(currentImg, (width,height))
lastImg = cv2.imread('output.png')
lastImg = cv2.resize(lastImg, (width,height))
# convert image to YCrCb and gray
currentGray = cv2.cvtColor(currentImg, cv2.COLOR_BGRA2GRAY)
lastGray = cv2.cvtColor(lastImg, cv2.COLOR_BGRA2GRAY)
currentImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2YCrCb)
lastImg = cv2.cvtColor(lastImg, cv2.COLOR_BGR2YCrCb)
# Split channel
currentY, currentCr, currentCb = cv2.split(currentImg)
lastY, lastCr, lastCb = cv2.split(lastImg)

# compute Level on Y, Cb and Cr channel
LevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterLevel))/256
#/256
levelCr = np.abs(cv2.filter2D(currentCr, cv2.CV_32F, FilterLevel))/256
#/256
levelCb = np.abs(cv2.filter2D(currentCb, cv2.CV_32F, FilterLevel))/256
#/256
# compute Edge on Y channel
EgdeY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdge))/36
#/36
# compute Spot on Y channel
SpotY = np.abs(cv2.filter2D(EgdeY, cv2.CV_32F, FilterSpot))/16
#/16
# Compute Edge+Level on Y channel
EdgeLevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeLevel1))/192 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeLevel2))/192
#/192
#/192
# Compute Spot+Level on Y channel
SpotLevelY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterSpotLevel1))/128 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterSpotLevel2))/128
#/128
#/128
# Compute Spot+Edge on Y channel
SpotEdgeY = np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeSpot1))/48 + np.abs(cv2.filter2D(currentY, cv2.CV_32F, FilterEdgeSpot2))/48
#/48
#/48


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
