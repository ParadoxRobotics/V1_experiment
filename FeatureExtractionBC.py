import math
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

# Size parameters
height = 128
width = 128

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
currentImg = cv2.imread('cur.png')
currentImg = cv2.resize(currentImg, (width,height))
lastImg = cv2.imread('ref.png')
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
FlowY = cv2.calcOpticalFlowFarneback(prev=lastGray, next=currentGray, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1, flags=0, flow=None)
# Plot feature
"""
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
"""
# concat representation
structHt = cv2.merge([LevelY, levelCr, levelCb, EgdeY, SpotY, EdgeLevelY, SpotLevelY, SpotEdgeY])
motionHt = np.abs(FlowY)

# 8x8x8=512 Spatial GIST feature (average and median)
FeatureVector = []
patchSize = 8
# Spatial Average GIST
for c in range(0, structHt.shape[2]):
    for i in range(0, patchSize):
        for j in range(0, patchSize):
            patch = structHt[int(i*structHt[:,:,c].shape[0]/patchSize):int(i*structHt[:,:,c].shape[0]/patchSize + structHt[:,:,c].shape[0]/patchSize), \
            int(j*structHt[:,:,c].shape[1]/patchSize):int(j*structHt[:,:,c].shape[1]/patchSize + structHt[:,:,c].shape[1]/patchSize), c]
            localMean = np.mean(patch)
            FeatureVector.append(localMean)

# Spatial Median GIST
for c in range(0, motionHt.shape[2]):
    for i in range(0, patchSize):
        for j in range(0, patchSize):
            patch = motionHt[int(i*motionHt[:,:,c].shape[0]/patchSize):int(i*motionHt[:,:,c].shape[0]/patchSize + motionHt[:,:,c].shape[0]/patchSize), \
            int(j*motionHt[:,:,c].shape[1]/patchSize):int(j*motionHt[:,:,c].shape[1]/patchSize + motionHt[:,:,c].shape[1]/patchSize), c]
            localMedian = np.median(patch)
            FeatureVector.append(localMedian)

# Rescale and Normalize Feature vector
print(FeatureVector)

# R-KNN classifier ?
