import math
import numpy as np
import cv2
import imutils
import copy
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt

# create spatial pyramid
def ImagePyramid(img, level):
    pyramid = []
    pyramid.append(img)
    for i in range(1, level):
        pyramid.append(cv2.pyrDown(pyramid[i-1]))
    return pyramid

# gabor filter bank generator
def gabor_filter_bank(kernelSize, wavelength, orientation):
    # standard gabor filter bank
    standardGaborBanks = []
    # compute parameter given the scale and wave wavelength
    lambdaValue = kernelSize*2/wavelength
    sigmaValue = lambdaValue * 0.8
    gammaValue = 0.3
    # compute orientation given the current scale
    for j in range(len(orientation)):
        theta = np.deg2rad(orientation[j]+90)
        # compute standard filters
        x, y = np.mgrid[:kernelSize, :kernelSize] - (kernelSize // 2)
        rotx = x * np.cos(theta) + y * np.sin(theta)
        roty = -x * np.sin(theta) + y * np.cos(theta)
        standardFilter = np.exp(-(rotx**2 + gammaValue**2 * roty**2) / (2 * sigmaValue ** 2))
        standardFilter *= np.cos(2 * np.pi * rotx / lambdaValue)
        standardFilter[np.sqrt(x**2 + y**2) > (kernelSize / 2)] = 0
        # normalize standard filter with mean=0 and std=1
        standardFilter = standardFilter-np.mean(standardFilter)
        standardFilter = standardFilter/np.std(standardFilter)
        # append
        standardGaborBanks.append(standardFilter.astype('float64'))
    # return value
    return standardGaborBanks

# Lateral geniculate nucleus processing
def LGN_processing(img, sigmaPos, sigmaNeg):
    # convert image to float and normalize it
    img = img.astype(np.float32) / 255.0
    # compute first gausian difference
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigmaPos, sigmaY=sigmaPos)
    num = img - blur
    # compute second gausian
    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=sigmaNeg, sigmaY=sigmaNeg)
    den = cv2.pow(blur, 0.5)
    img = num / den
    return img

# Simple log-gabor cell processing
def simple_cell_processing(img, filterBanks):
    SCG = []
    for o in range(0, len(filterBanks)):
        featureMap = cv2.filter2D(img, -1, filterBanks[o])
        # apply ReLU function
        featureMap[featureMap < 0] = 0
        # append
        SCG.append(featureMap.astype('float64'))
    # return feature maps
    return SCG

# Complex log-gabor cell processing
def complex_cell_processing(img, filterBanks, filterBanksDephase):
    CCG = []
    for o in range(0, len(filterBanks)):
        featureMapInit = cv2.filter2D(img, -1, filterBanks[o])
        featureMapDephase = cv2.filter2D(img, -1, filterBanksDephase[o])
        # compute complexe cell
        featureMap = (np.sqrt((featureMapInit**2+featureMapDephase**2))/np.sqrt(2))
        # append
        CCG.append(featureMap.astype('float64'))
    # return feature maps
    return CCG

# Simple color processing
def color_channel_processing(img):
    CC = []
    # separate BGR Channel
    B,G,R = cv2.split(img)
    # basic global color opponnency
    CR = R-((G+B)/2)
    CG = G-((R+B)/2)
    CB = B-((R+G)/2)
    CY = ((R+G)/2)-(np.abs(R-G)/2)-B
    CRG = CR-CG
    CGR = CG-CR
    CBY = CB-CY
    CYB = CY-CB
    CC.append(CRG)
    CC.append(CGR)
    CC.append(CBY)
    CC.append(CYB)
    return CC

# get state
img = cv2.imread("/home/cyborg67/Bureau/lenna.png")
img = cv2.resize(img, (256,256))
plt.matshow(img)
plt.show()

# create spatial pyramid
imgPyr = ImagePyramid(img, 4)
# normalize color image pyramid
colorImgPyr = []
for l in range(0,len(imgPyr)):
    colorImg = imgPyr[l] - np.mean(imgPyr[l])
    colorImg = colorImg / np.std(colorImg)
    colorImgPyr.append(colorImg)
    plt.matshow(colorImg)
    plt.show()
# normalize gray image
grayImgPyr = []
for l in range(0,len(imgPyr)):
    grayImg = cv2.cvtColor(imgPyr[l], cv2.COLOR_BGR2GRAY) - np.mean(cv2.cvtColor(imgPyr[l], cv2.COLOR_BGR2GRAY))
    grayImg = grayImg / np.std(grayImg)
    grayImgPyr.append(grayImg)
    plt.matshow(grayImg)
    plt.show()
# LGN processing
LGNImgPyr = []
for l in range(0,len(grayImgPyr)):
    LGNImg = LGN_processing(grayImgPyr[l], sigmaPos=2, sigmaNeg=20)
    LGNImgPyr.append(LGNImg)
    plt.matshow(LGNImg)
    plt.show()

# create gabor banks
standardGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0,45,90,135])
dephasedGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0+90,45+90,90+90,135+90])
# simple gabor banks
print("Simple Gabor cell")
for i in range(0, len(standardGaborBanks)):
    plt.matshow(standardGaborBanks[i])
    plt.show()
# complexe gabor banks
print("dephased Gabor cell")
for i in range(0, len(dephasedGaborBanks)):
    plt.matshow(dephasedGaborBanks[i])
    plt.show()

# compute simple cell response at each scale of the image pyramid
simpleCellPyr = []
for l in range(0,len(grayImgPyr)):
    simpleCellPyr.append(simple_cell_processing(grayImgPyr[l], standardGaborBanks))
# compute complexe cell response at each scale of the image pyramid
complexeCellPyr = []
for l in range(0,len(grayImgPyr)):
    complexeCellPyr.append(complex_cell_processing(grayImgPyr[l], standardGaborBanks, dephasedGaborBanks))
# simple gabor response
print("Simple Gabor response")
for l in range(0, len(simpleCellPyr)):
    for o in range(0, len(simpleCellPyr[0])):
        plt.matshow(simpleCellPyr[l][o])
        plt.show()
# complexe gabor response
print("Complexe Gabor response")
for l in range(0, len(complexeCellPyr)):
    for o in range(0, len(complexeCellPyr[0])):
        plt.matshow(complexeCellPyr[l][o])
        plt.show()

# Compute color opponent response at each scale of the image pyramid
colorOpponentPyr = []
for l in range(0,len(colorImgPyr)):
    colorOpponentPyr.append(color_channel_processing(colorImgPyr[l]))
# color opponent response
print("Color opponent response")
for l in range(0, len(colorOpponentPyr)):
    for c in range(0, len(colorOpponentPyr[0])):
        plt.matshow(colorOpponentPyr[l][c])
        plt.show()
