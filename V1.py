import math
import numpy as np
import cv2
import imutils
import copy
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt

# Log-gabor filter bank generator
def gabor_filter_bank(kernelSize, wavelength, orientation):
    # standard gabor filter bank
    standardGaborBanks = []
    # positive gabor filter bank
    positiveGaborBanks = []
    # negative gabor filter bank
    negativeGaborBanks = []
    # create filter banks
    for i in range(len(kernelSize)):
        # temp filter
        standardGabor = []
        positiveGabor = []
        negativeGabor = []
        # compute parameter given the scale and wave wavelength
        lambdaValue = kernelSize[i]*2/wavelength[i]
        sigmaValue = lambdaValue * 0.8
        gammaValue = 0.3
        # compute orientation given the current scale
        for j in range(len(orientation)):
            theta = np.deg2rad(orientation[j]+90)
            # compute standard filters
            x, y = np.mgrid[:kernelSize[i], :kernelSize[i]] - (kernelSize[i] // 2)
            rotx = x * np.cos(theta) + y * np.sin(theta)
            roty = -x * np.sin(theta) + y * np.cos(theta)
            standardFilter = np.exp(-(rotx**2 + gammaValue**2 * roty**2) / (2 * sigmaValue ** 2))
            standardFilter *= np.cos(2 * np.pi * rotx / lambdaValue)
            standardFilter[np.sqrt(x**2 + y**2) > (kernelSize[i] / 2)] = 0
            # normalize standard filter with mean=0 and std=1
            standardFilter = standardFilter-np.mean(standardFilter)
            standardFilter = standardFilter/np.std(standardFilter)
            # compute positive filters
            positiveFilter = np.copy(standardFilter)
            positiveFilter[positiveFilter<0] = 0
            # compute negative filters
            negativeFilter = np.copy(standardFilter)
            negativeFilter[negativeFilter>0] = 0
            # append
            standardGabor.append(standardFilter)
            positiveGabor.append(positiveFilter)
            negativeGabor.append(negativeFilter)
        # append
        standardGaborBanks.append(standardGabor)
        positiveGaborBanks.append(positiveGabor)
        negativeGaborBanks.append(negativeGabor)
    # return value
    return standardGaborBanks, positiveGaborBanks, negativeGaborBanks

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
    for s in range(0, len(filterBanks)):
        for o in range(0, len(filterBanks[0])):
            featureMap = cv2.filter2D(img, -1, filterBanks[s][o])
            # apply ReLU function
            featureMap[featureMap < 0] = 0
            # append
            SCG.append(featureMap.astype('float64'))
    # return feature maps
    return SCG

# Complex log-gabor cell processing
def complex_cell_processing(img, filterBanks, filterBanksDephase):
    CCG = []
    for s in range(0, len(filterBanks)):
        for o in range(0, len(filterBanks[0])):
            featureMapInit = cv2.filter2D(img, -1, filterBanks[s][o])
            featureMapDephase = cv2.filter2D(img, -1, filterBanksDephase[s][o])
            # compute complexe cell
            featureMap = (np.sqrt((featureMapInit**2+featureMapDephase**2))/np.sqrt(2))
            # append
            CCG.append(featureMap.astype('float64'))
    # return feature maps
    return CCG

# Color oppponent cell processing
def color_cell_processing(img, positiveGaborBanks, negativeGaborBanks, weights):
    # simple oppponent and double opponent cells
    SO = []
    DO = []
    # current bank use for each channel given the weights
    RgaborBank = None
    GgaborBank = None
    BgaborBank = None
    # weights
    WR = 0.0
    WG = 0.0
    WB = 0.0
    # separate BGR Channel
    B,G,R = cv2.split(img)
    # for every weight vector compute the SO and DO cell
    for w in range(len(weights)):
        # get weight and gabor config for each channel
        if(weights[w][0] > 0):
            RgaborBank = positiveGaborBanks
        else:
            RgaborBank = negativeGaborBanks
        if(weights[w][1] > 0):
            GgaborBank = positiveGaborBanks
        else:
            GgaborBank = negativeGaborBanks
        if(weights[w][2] > 0):
            BgaborBank = positiveGaborBanks
        else:
            BgaborBank = negativeGaborBanks
        # compute gabor filter
        for s in range(0, len(positiveGaborBanks)):
            for o in range(0, len(positiveGaborBanks[0])):
                Rgabor = cv2.filter2D(R, -1, RgaborBank[s][o])
                Ggabor = cv2.filter2D(G, -1, GgaborBank[s][o])
                Bgabor = cv2.filter2D(B, -1, BgaborBank[s][o])
                # weighted sum
                SOfeature = weights[w][0]*Rgabor + weights[w][1]*Ggabor + weights[w][2]*Bgabor
                # apply ReLU function
                SOfeature[SOfeature < 0] = 0
                # divisive normalization


    return SO, DO

# Hierarchical center and surround processing

# get state
img = cv2.imread("/home/cyborg67/Bureau/tf.jpg")
img = cv2.resize(img, (256,256))
# LGN processing
colorImg = LGN_processing(img, sigmaPos=2, sigmaNeg=20)
plt.matshow(colorImg)
plt.show()
grayImg = LGN_processing(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigmaPos=2, sigmaNeg=20)
plt.matshow(grayImg)
plt.show()
# create gabor banks
standardGaborBanks, positiveGaborBanks, negativeGaborBanks = gabor_filter_bank(kernelSize=[7,9,11,13], wavelength=[4,3.95,3.9,3.85], orientation=[0,45,90,135])
standardGaborBanksDephase, positiveGaborBanksDephase, negativeGaborBanksDephase = gabor_filter_bank(kernelSize=[7,9,11,13], wavelength=[4,3.95,3.9,3.85], orientation=[0+90,45+90,90+90,135+90])
# color oppponent cell weight and value
colorWeight = [[1/np.sqrt(2), -1/np.sqrt(2), 0.0],
               [-1/np.sqrt(2), 1/np.sqrt(2), 0.0],
               [2/np.sqrt(6),-1/np.sqrt(6),-1/np.sqrt(6)],
               [-2/np.sqrt(6), 1/np.sqrt(6), 1/np.sqrt(6)],
               [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
               [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
               [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
               [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]]

# compute simple cell
SC = simple_cell_processing(grayImg, standardGaborBanks)
# compute complexe cell
CC = complex_cell_processing(grayImg, standardGaborBanks, standardGaborBanksDephase)

for i in range(0, len(SC)):
    plt.matshow(SC[i])
    plt.show()

for i in range(0, len(CC)):
    plt.matshow(CC[i])
    plt.show()

"""
Value glossary :
S1(size=7, wavelength=4),
S1(size=9, wavelength=3.95),
S1(size=11, wavelength=3.9),
S1(size=13, wavelength=3.85),
S1(size=15, wavelength=3.8),
S1(size=17, wavelength=3.75),
S1(size=19, wavelength=3.7),
S1(size=21, wavelength=3.65),
S1(size=23, wavelength=3.6),
S1(size=25, wavelength=3.55),
S1(size=27, wavelength=3.5),
S1(size=29, wavelength=3.45),
S1(size=31, wavelength=3.4),
S1(size=33, wavelength=3.35),
S1(size=35, wavelength=3.3),
S1(size=37, wavelength=3.25),
"""
