import math
import numpy as np
from scipy.special import spherical_jn
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

# Gaussian center and surround processing
def center_and_surround(img, kernelSize, sigmaI, sigmaO):
    # compute inner gaussian
    innerGaussian = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigmaI, sigmaY=sigmaI)
    # compute outer gaussian
    outerGaussian = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigmaO, sigmaY=sigmaO)
    # compute ON center and surround response
    CSON = innerGaussian - outerGaussian
    # compute OFF center and surround response
    CSOFF = -innerGaussian + outerGaussian
    return CSON, CSOFF

# generate center and surround filter
def center_and_surround_filter(kernelSize, sigmaI, sigmaO):
    CSON = []
    CSOFF = []
    # compute filter
    x, y = np.mgrid[:kernelSize, :kernelSize] - (kernelSize // 2)
    CSON = (1/(2*np.pi*sigmaI**2))*np.exp(-((x**2+y**2)/(2*sigmaI**2))) - (1/(2*np.pi*sigmaO**2))*np.exp(-((x**2+y**2)/(2*sigmaO**2)))
    CSOFF = -(1/(2*np.pi*sigmaI**2))*np.exp(-((x**2+y**2)/(2*sigmaI**2))) + (1/(2*np.pi*sigmaO**2))*np.exp(-((x**2+y**2)/(2*sigmaO**2)))
    return CSON, CSOFF

# Von Mises distribition filter
def von_mises_filter(kernelSize, radius, orientation):
    VMD = []
    sigma = radius/2
    # create kernel grid
    for o in range(0, len(orientation)):
        orient = np.deg2rad(orientation[o]+90)
        # compute filter
        x, y = np.mgrid[:kernelSize, :kernelSize] - (kernelSize // 2)
        R = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x)
        VMF = np.exp(radius*np.cos(theta-(orient)))/np.i0(R-radius)
        # normalize
        VMF = VMF/np.max(np.max(VMF))
        # append
        VMD.append(VMF)
    return VMD

# compute von Mise filter over an image
def von_mises_processing(img, vonMisesBanks):
    VMF = []
    for o in range(0, len(vonMisesBanks)):
        VMF.append(cv2.filter2D(img, -1, vonMisesBanks[o]))
    return VMF

# sum von mise reponse over a target scale and orient
def sumVonMiseScale(vonMiseReponsePyr, scale, orientation):
    # get the shape of the summed map
    targetSize = vonMiseReponsePyr[scale][orientation].shape[0]
    # init sum map
    sum = np.empty([targetSize, targetSize])
    if(scale==0):
        sum = (1/2)*cv2.resize(vonMiseReponsePyr[scale][orientation], (targetSize, targetSize))
        return sum
    else:
        sum = (1/2)*cv2.resize(vonMiseReponsePyr[0][orientation], (targetSize, targetSize))
        for s in range(1,scale):
            sum += (1/2**s+1)*cv2.resize(vonMiseReponsePyr[s][orientation], (targetSize, targetSize))
    return sum

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
        theta = np.deg2rad(orientation[j])
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

# border ownership processing
def border_ownership(featurePyr, featurePyrON, featurePyrOFF, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks):
    # compute complexe cell reponse over all orientation and scale
    complexeCellPyr = []
    for l in range(0,len(grayImgPyr)):
        complexeCellPyr.append(complex_cell_processing(featurePyr[l], standardGaborBanks, dephasedGaborBanks))
    # compute von Mises filter response over all orientation and scale
    vonMisesON = []
    dephasedVonMisesON = []
    vonMiseOFF = []
    dephasedVonMisesOFF = []
    for l in range(0,len(grayImgPyr)):
        vonMisesON.append(von_mises_processing(featurePyrON[l], vonMisesBanks))
        dephasedVonMisesON.append(von_mises_processing(featurePyrON[l], dephasedVonMisesBanks))
        vonMiseOFF.append(von_mises_processing(featurePyrOFF[l], vonMisesBanks))
        dephasedVonMisesOFF.append(von_mises_processing(featurePyrOFF[l], dephasedVonMisesBanks))
    # compute border cell ON and OFF
    B = []
    for level in range(0, len(vonMisesON)):
        BO = []
        for orient in range(0, len(vonMisesON[0])):
            BL = complexeCellPyr[level][orient]*(1+sumVonMiseScale(dephasedVonMisesON, level, orient)-1*sumVonMiseScale(vonMisesON, level, orient))
            BD = complexeCellPyr[level][orient]*(1+sumVonMiseScale(dephasedVonMisesOFF, level, orient)-1*sumVonMiseScale(vonMiseOFF, level, orient))
            BO.append(BL-BD)
        # append value
        B.append(BO)
    return B

# grouping using simplified gestalt principle
def grouping_border_ownership(BO, BOD, vonMises):
    G = []
    for level in range(0, len(BO)):
        GO = []
        sumGO = np.empty([BO[l][0].shape[0], BO[l][0].shape[0]])
        for orient in range(0, len(BO[0])):
            if (orient==0):
                sumGO = cv2.filter2D(BO[level][orient]-BOD[level][orient], -1, vonMises[orient])
            else:
                sumGO += cv2.filter2D(BO[level][orient]-BOD[level][orient], -1, vonMises[orient])
        # append summed orientatio border
        G.append(sumGO)
    return G

# Flicker motion processing
def flicker_processing(imgCur, imgPrev):
    return cv2.absdiff(imgCur, imgPrev)

# Motion orientation processing
def motion_processing(orientCur, prevOrient):
    return cv2.absdiff(orientCur, prevOrient)

# get state
img = cv2.imread("/home/main/Bureau/car2.jpg")
img = cv2.resize(img, (512,512))
plt.matshow(img)
plt.show()
prevImg = cv2.imread("/home/main/Bureau/car1.jpg")
prevImg = cv2.resize(prevImg, (512,512))
plt.matshow(prevImg)
plt.show()

# create spatial pyramid
imgPyr = ImagePyramid(img, 3)
prevImgPyr = ImagePyramid(prevImg, 3)
# normalize color image pyramid
colorImgPyr = []
for l in range(0,len(imgPyr)):
    colorImg = imgPyr[l] - np.mean(imgPyr[l])
    colorImg = colorImg / np.std(colorImg)
    colorImgPyr.append(colorImg)

# normalize gray image
grayImgPyr = []
for l in range(0,len(imgPyr)):
    grayImg = cv2.cvtColor(imgPyr[l], cv2.COLOR_BGR2GRAY) - np.mean(cv2.cvtColor(imgPyr[l], cv2.COLOR_BGR2GRAY))
    grayImg = grayImg / np.std(grayImg)
    grayImgPyr.append(grayImg)

# normalize previous gray image
prevGrayImgPyr = []
for l in range(0,len(prevImgPyr)):
    prevGrayImg = cv2.cvtColor(prevImgPyr[l], cv2.COLOR_BGR2GRAY) - np.mean(cv2.cvtColor(prevImgPyr[l], cv2.COLOR_BGR2GRAY))
    prevGrayImg = prevGrayImg / np.std(prevGrayImg)
    prevGrayImgPyr.append(prevGrayImg)

# LGN processing
LGNImgPyr = []
for l in range(0,len(grayImgPyr)):
    LGNImg = LGN_processing(grayImgPyr[l], sigmaPos=2, sigmaNeg=20)
    LGNImgPyr.append(LGNImg)

# create gabor banks
standardGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0,45,90,135])
dephasedGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[90,135,180,225])

standard2GaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0+180,45+180,90+180,135+180])
dephased2GaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[90+180,135+180,180+180,225+180])

# compute simple cell response at each scale of the image pyramid
simpleCellPyr = []
for l in range(0,len(grayImgPyr)):
    simpleCellPyr.append(simple_cell_processing(grayImgPyr[l], standardGaborBanks))
# compute simple cell response at each scale of the image pyramid from the previous state
prevSimpleCellPyr = []
for l in range(0,len(prevGrayImgPyr)):
    prevSimpleCellPyr.append(simple_cell_processing(prevGrayImgPyr[l], standardGaborBanks))

# Compute color opponent response at each scale of the image pyramid
colorOpponentPyr = []
for l in range(0,len(colorImgPyr)):
    colorOpponentPyr.append(color_channel_processing(colorImgPyr[l]))

# compute flicker response at each scale of the spatial pyramid
flickerPyr = []
for l in range(0, len(grayImgPyr)):
    flickerPyr.append(flicker_processing(grayImgPyr[l], prevGrayImgPyr[l]))

# compute motion response from spatial difference from 2 state image pyramid
motionPyr = []
for l in range(0, len(simpleCellPyr)):
    orientDiff = []
    for o in range(0, len(simpleCellPyr[0])):
        orientDiff.append(motion_processing(simpleCellPyr[l][o], prevSimpleCellPyr[l][o]))
    # append motion
    motionPyr.append(orientDiff)

# compute center and surround on edge, color and motion
grayON = []
grayOFF = []
simpleCellON = []
simpleCellOFF = []
colorOpponentON = []
colorOpponentOFF = []
flickerCellON = []
flickerCellOFF = []
motionCellON = []
motionCellOFF = []
# compute at all level in the pyramid
for l in range(0, len(colorImgPyr)):
    # basic color
    GON, GOFF = center_and_surround(grayImgPyr[l], 5, 0.9, 2.7)
    grayON.append(GON)
    grayOFF.append(GOFF)

    # basic color opponent
    COON = []
    COOFF = []
    # CAS on color
    for c in range(0, len(colorOpponentPyr[0])):
        COCON, COCOFF = center_and_surround(colorOpponentPyr[l][c], 5, 0.9, 2.7)
        COON.append(COCON)
        COOFF.append(COCOFF)
    # append for all scale
    colorOpponentON.append(COON)
    colorOpponentOFF.append(COOFF)

    # simple cell
    SCON = []
    SCOFF = []
    # CAS on simple cell
    for o in range(0, len(simpleCellPyr[0])):
        SCOON, SCOOFF = center_and_surround(simpleCellPyr[l][o], 5, 0.9, 2.7)
        # append current scale
        SCON.append(SCOON)
        SCOFF.append(SCOOFF)
    # append for all scale
    simpleCellON.append(SCON)
    simpleCellOFF.append(SCOFF)

    # flicker cell
    FCON, FCOFF = center_and_surround(flickerPyr[l], 5, 0.9, 2.7)
    flickerCellON.append(FCON)
    flickerCellOFF.append(FCOFF)

    # motion cell
    MCON = []
    MCOFF = []
    # CAS on simple cell
    for m in range(0, len(motionPyr[0])):
        MOON, MOOFF = center_and_surround(motionPyr[l][m], 5, 0.9, 2.7)
        MCON.append(MOON)
        MCOFF.append(MOOFF)
    # append for all scale
    motionCellON.append(MCON)
    motionCellOFF.append(MCOFF)

# create Von mises filter banks
vonMisesBanks = von_mises_filter(kernelSize=7, radius=2.7, orientation=[0,45,90,135])
dephasedVonMisesBanks = von_mises_filter(kernelSize=7, radius=2.7, orientation=[180,45+180,90+180,135+180])
dephased2VonMisesBanks = von_mises_filter(kernelSize=7, radius=2.7, orientation=[360,45+360,90+360,135+360])

# reshape pyramid (TODO : better structure at start for easier computation)
RGON = []
GRON = []
RGOFF = []
GROFF = []
BYON = []
YBON = []
BYOFF = []
BYON = []
for l in range(0, colorOpponentON):
    RGON.append(colorOpponentON[l][0])
    GRON.append(colorOpponentON[l][1])
    RGOFF.append(colorOpponentOFF[l][0])
    GROFF.append(colorOpponentOFF[l][1])
    BYON.append(colorOpponentON[l][2])
    YBON.append(colorOpponentON[l][3])
    BYOFF.append(colorOpponentOFF[l][2])
    YBOFF.append(colorOpponentOFF[l][3])

G0ON = []
G45ON = []
G90ON = []
G135ON = []
G0OFF = []
G45OFF = []
G90OFF = []
G135OFF = []
for l in range(0, simpleCellON):
    G0ON.append(simpleCellON[l][0])
    G45ON.append(simpleCellON[l][1])
    G0OFF.append(simpleCellOFF[l][0])
    G45OFF.append(simpleCellOFF[l][1])
    G90ON.append(simpleCellON[l][2])
    G135ON.append(simpleCellON[l][3])
    G90OFF.append(simpleCellOFF[l][2])
    G135OFF.append(simpleCellOFF[l][3])

# compute border ownership for gray pyramid
grayPyrBO = border_ownership(grayImgPyr, grayON, grayOFF, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks)
DephasedGrayPyrBO = border_ownership(grayImgPyr, grayON, grayOFF, standard2GaborBanks, dephased2GaborBanks, dephasedVonMisesBanks, dephased2VonMisesBanks)
# gestalt grouping
GroupingGrayPyr = grouping_border_ownership(grayPyrBO, DephasedGrayPyrBO, vonMisesBanks)

for l in range(0, len(GroupingGrayPyr)):
    plt.matshow(GroupingGrayPyr[l])
    plt.show()
