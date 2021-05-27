import math
import numpy as np
import cv2
import imutils
import copy
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt

# Create spatial pyramid
def ImagePyramid(img, level):
    pyramid = []
    pyramid.append(img)
    for i in range(1, level):
        pyramid.append(cv2.pyrDown(pyramid[i-1]))
    return pyramid

def mean_std_normalization(imgPyr):
    NPyr = []
    for l in range(0, len(imgPyr)):
        normalizedPyr = imgPyr[l]-np.mean(imgPyr[l])
        normalizedPyr = normalizedPyr/np.std(normalizedPyr)
        NPyr.append(normalizedPyr)
    return NPyr

def intensity_pyramid(imgPyr):
    intPyr = []
    for l in range(0, len(imgPyr)):
        intPyr.append(cv2.cvtColor(imgPyr[l], cv2.COLOR_BGR2GRAY))
    return intPyr

# Lateral geniculate nucleus processing over an image/feature pyramid
def LGN_processing(imgPyr, sigmaPos, sigmaNeg):
    LGNPyr = []
    for l in range(0, len(imgPyr)):
        # convert image to float and normalize it
        img = imgPyr[l].astype(np.float32) / 255.0
        # compute first gausian difference
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigmaPos, sigmaY=sigmaPos)
        num = img - blur
        # compute second gausian
        blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=sigmaNeg, sigmaY=sigmaNeg)
        den = cv2.pow(blur, 0.5)
        img = num / den
        LGNPyr.append(img)
    return LGNPyr

# Gaussian center and surround processing over an image/feature pyramid
def center_and_surround_single(imgPyr, kernelSize, sigmaI, sigmaO):
    CCPyrON = []
    CCPyrOFF = []
    for l in range(0, len(imgPyr)):
        innerGaussian = cv2.GaussianBlur(imgPyr[l], (kernelSize, kernelSize), sigmaX=sigmaI, sigmaY=sigmaI)
        outerGaussian = cv2.GaussianBlur(imgPyr[l], (kernelSize, kernelSize), sigmaX=sigmaO, sigmaY=sigmaO)
        CSON = innerGaussian - outerGaussian
        CSON[CSON < 0] = 0
        CSOFF = -innerGaussian + outerGaussian
        CSOFF[CSOFF < 0] = 0
        CCPyrON.append(CSON)
        CCPyrOFF.append(CSOFF)
    return CCPyrON, CCPyrOFF

def center_and_surround_feature(imgPyr, kernelSize, sigmaI, sigmaO):
    CCPyrON = []
    CCPyrOFF = []
    for o in range(0, len(imgPyr)):
        CCPyrFeatureON = []
        CCPyrFeatureOFF = []
        for l in range(0, len(imgPyr[0])):
            innerGaussian = cv2.GaussianBlur(imgPyr[o][l], (kernelSize, kernelSize), sigmaX=sigmaI, sigmaY=sigmaI)
            outerGaussian = cv2.GaussianBlur(imgPyr[o][l], (kernelSize, kernelSize), sigmaX=sigmaO, sigmaY=sigmaO)
            CSON = innerGaussian - outerGaussian
            CSON[CSON < 0] = 0
            CSOFF = -innerGaussian + outerGaussian
            CSOFF[CSOFF < 0] = 0
            CCPyrFeatureON.append(CSON)
            CCPyrFeatureOFF.append(CSOFF)
        CCPyrON.append(CCPyrFeatureON)
        CCPyrOFF.append(CCPyrFeatureOFF)
    return CCPyrON, CCPyrOFF

# Circular Von Mises filter
def von_mises_filter_bank(kernelSize, radius, orientation):
    VMB = []
    sigma = radius/2
    for o in range(0, len(orientation)):
        orient = np.deg2rad(orientation[o]+90)
        # compute filter
        x, y = np.mgrid[:kernelSize, :kernelSize] - (kernelSize // 2)
        R = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x)
        VMF = np.exp(radius*np.cos(theta-(orient)))/np.i0(R-radius)
        VMF = VMF/np.max(np.max(VMF))
        VMB.append(VMF.astype('float64'))
    # return Von Mise filter bank
    return VMB

# Compute von Mise filter over an image/feature pyramid
def von_mises_processing(imgPyr, vonMisesBank):
    VMF = []
    for f in range(0, len(vonMisesBanks)):
        featurePyramid = []
        for l in range(0, len(imgPyr)):
            featurePyramid.append(cv2.filter2D(imgPyr[l], -1, vonMisesBank[f]).astype('float64'))
        VMF.append(featurePyramid)
    # return filter feature pyramid at different orientation ans scale
    return VMF

# Sum von mise reponse over a target scale
def sum_von_mise_scale(vonMiseReponsePyr, scale, orientation):
    # get the shape of the summed map
    targetSize = vonMiseReponsePyr[orientation][scale].shape[0]
    # init sum map
    sum = np.empty([targetSize, targetSize])
    if(scale==0):
        sum = (1/2)*cv2.resize(vonMiseReponsePyr[orientation][scale], (targetSize, targetSize))
        return sum
    else:
        sum = (1/2)*cv2.resize(vonMiseReponsePyr[0][scale], (targetSize, targetSize))
        for s in range(1,scale):
            sum += (1/2**s+1)*cv2.resize(vonMiseReponsePyr[orientation][s], (targetSize, targetSize))
    # return the normalized sum
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
    # return gabor bank
    return standardGaborBanks

# Compute Simple Gabor filter over an image/feature pyramid
def simple_cell_processing(imgPyr, filterBanks):
    SCG = []
    for o in range(0, len(filterBanks)):
        featurePyramid = []
        for l in range(0, len(imgPyr)):
            featureMap = cv2.filter2D(imgPyr[l], -1, filterBanks[o])
            # apply ReLU function
            featureMap[featureMap < 0] = 0
            # append
            featurePyramid.append(featureMap.astype('float64'))
        SCG.append(featurePyramid)
    # return feature maps
    return SCG

# Compute Complex Gabor filter over an image/feature pyramid
def complex_cell_processing(imgPyr, filterBanks, filterBanksDephase):
    CCG = []
    for o in range(0, len(filterBanks)):
        featurePyramid = []
        for l in range(0, len(imgPyr)):
            featureMapInit = cv2.filter2D(imgPyr[l], -1, filterBanks[o])
            featureMapDephase = cv2.filter2D(imgPyr[l], -1, filterBanksDephase[o])
            # compute complexe cell
            featureMap = (np.sqrt((featureMapInit**2+featureMapDephase**2))/np.sqrt(2))
            # append
            featurePyramid.append(featureMap.astype('float64'))
        CCG.append(featurePyramid)
    # return feature maps
    return CCG

# Compute color opponent feature map over an image/feature pyramid
def color_opponent_processing(imgPyr):
    CC = []
    RGPyramid = []
    GRPyramid = []
    BYPyramid = []
    YBPyramid = []
    for l in range(0, len(imgPyr)):
        # separate BGR Channel
        B,G,R = cv2.split(imgPyr[l])
        # basic global color opponnency
        CR = R-((G+B)/2)
        CG = G-((R+B)/2)
        CB = B-((R+G)/2)
        CY = ((R+G)/2)-(np.abs(R-G)/2)-B
        CRG = CR-CG
        CGR = CG-CR
        CBY = CB-CY
        CYB = CY-CB
        # append pyramid
        RGPyramid.append(CRG)
        GRPyramid.append(CGR)
        BYPyramid.append(CBY)
        YBPyramid.append(CYB)
    # append feature map pyramid
    CC.append(RGPyramid)
    CC.append(GRPyramid)
    CC.append(BYPyramid)
    CC.append(YBPyramid)
    return CC

# Compute Flicker motion feature map over an image/feature pyramid
def flicker_processing(imgPyrCur, imgPyrRef):
    FlickerPyr = []
    for l in range(0, len(imgPyrCur)):
        FlickerPyr.append(cv2.absdiff(imgPyrCur[l], imgPyrRef[l]))
    return FlickerPyr

# border ownership processing
def border_ownership(featurePyr, featurePyrON, featurePyrOFF, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks):
    # compute complexe cell reponse over all orientation and scale
    complexeCellPyr = complex_cell_processing(featurePyr, standardGaborBanks, dephasedGaborBanks)
    # compute von Mises filter response over all orientation and scale
    vonMisesON = von_mises_processing(featurePyrON, vonMisesBanks)
    dephasedVonMisesON = von_mises_processing(featurePyrON, dephasedVonMisesBanks)
    vonMiseOFF = von_mises_processing(featurePyrOFF, vonMisesBanks)
    dephasedVonMisesOFF = von_mises_processing(featurePyrOFF, dephasedVonMisesBanks)
    # compute border cell ON and OFF
    B = []
    for o in range(0, len(vonMisesON)):
        BO = []
        for l in range(0, len(vonMisesON[0])):
            BL = complexeCellPyr[o][l]*(1+sum_von_mise_scale(dephasedVonMisesON, l, o)-1*sum_von_mise_scale(vonMisesON, l, o))
            BD = complexeCellPyr[o][l]*(1+sum_von_mise_scale(dephasedVonMisesOFF, l, o)-1*sum_von_mise_scale(vonMiseOFF, l, o))
            BAdd = BL+BD
            BAdd[BAdd < 0] = 0
            BO.append(BAdd)
        # append value
        B.append(BO)
    return B

# Argmax Border ownership orientation pyramid
def argmax_border_ownership(BO, BOD):
    BK = []
    BDiff = []
    # compute difference
    for o in range(0, len(BO)):
        featurePyr = []
        for l in range(0, len(BO[0])):
            featurePyr.append(BO[o][l]-BOD[o][l])
        # append pyramid
        BDiff.append(featurePyr)
    # compute maximum
    for l in range(0, len(BDiff[0])):
        featureMap = []
        for o in range(0, len(BDiff)):
            featureMap.append(BDiff[o][l])
        # compute maximum over border orientation
        BK.append(np.maximum.reduce(featureMap))
    return BK

# Compute grouping using simplified Gestalt principle
def grouping_border_ownership(BO, BOD, vonMises):
    # compute Argmax border
    BK = argmax_border_ownership(BO, BOD)
    # compute grouping
    G = []
    for l in range(0, len(BO[0])):
        GO = []
        sumGO = np.empty([BO[0][l].shape[0], BO[0][l].shape[0]])
        for o in range(0, len(BO)):
            maskBK = np.zeros((BO[o][l].shape[0], BO[o][l].shape[0]))
            if (o==0):
                # compute mask
                maskBK[BO[o][l]>BK[l]] = 1
                sumGO = maskBK*cv2.filter2D(BO[o][l]-2*BOD[o][l], -1, vonMises[o])
            else:
                # compute mask
                maskBK[BO[o][l]>BK[l]] = 1
                sumGO += maskBK*cv2.filter2D(BO[o][l]-2*BOD[o][l], -1, vonMises[o])
        # append summed orientation border
        G.append(sumGO)
    return G

# get state
img = cv2.imread("/home/cyborg67/Bureau/cur.png")
img = cv2.resize(img, (640,640))
plt.matshow(img)
plt.show()
prevImg = cv2.imread("/home/cyborg67/Bureau/ref.png")
prevImg = cv2.resize(prevImg, (640,640))
plt.matshow(prevImg)
plt.show()

# create spatial pyramid
refImgPyr = ImagePyramid(img, 3)
curImgPyr = ImagePyramid(prevImg, 3)

# compute gabor filter bank
standardGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0,45,90,135])
dephasedGaborBanks = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[90,135,180,225])
standardGaborBanksPI = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[0+180,45+180,90+180,135+180])
dephasedGaborBanksPI = gabor_filter_bank(kernelSize=7, wavelength=4, orientation=[90+180,135+180,180+180,225+180])
# compute Von Mise filter bank
vonMisesBanks = von_mises_filter_bank(kernelSize=7, radius=2.7, orientation=[0,45,90,135])
dephasedVonMisesBanks = von_mises_filter_bank(kernelSize=7, radius=2.7, orientation=[180,45+180,90+180,135+180])
dephasedVonMisesBanksPI = von_mises_filter_bank(kernelSize=7, radius=2.7, orientation=[360,45+360,90+360,135+360])

# compute normalized gray pyramid
refGrayPyr = mean_std_normalization(imgPyr=intensity_pyramid(imgPyr=refImgPyr))
curGrayPyr = mean_std_normalization(imgPyr=intensity_pyramid(imgPyr=curImgPyr))
# compute color opponent pyramid
colorPyr = color_opponent_processing(imgPyr=mean_std_normalization(imgPyr=refImgPyr))
# compute LGN pyramid
refLGNPyr = LGN_processing(imgPyr=refGrayPyr, sigmaPos=2, sigmaNeg=20)
curLGNPyr = LGN_processing(imgPyr=curGrayPyr, sigmaPos=2, sigmaNeg=20)
# compute gabor pyramid
simplePyr = simple_cell_processing(imgPyr=curLGNPyr, filterBanks=standardGaborBanks)
# compute flicker pyramid
flickerPyr = flicker_processing(imgPyrCur=curGrayPyr, imgPyrRef=refGrayPyr)

# compute ON/OFF gray cell pyramid
refGrayONPyr, refGrayOFFPyr = center_and_surround_single(imgPyr=refGrayPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
curGrayONPyr, curGrayOFFPyr = center_and_surround_single(imgPyr=curGrayPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
# compute ON/OFF LGN cell pyramid
refLGNONPyr, refLGNOFFPyr = center_and_surround_single(imgPyr=refLGNPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
curLGNONPyr, curLGNOFFPyr = center_and_surround_single(imgPyr=curLGNPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
# compute ON/OFF flicker cell pyramid
flickerONPyr, flickerOFFPyr = center_and_surround_single(imgPyr=flickerPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
# compute ON/OFF color opponent cell pyramid
colorONPyr, colorOFFPyr = center_and_surround_feature(imgPyr=colorPyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)
# compute ON/OFF simple cell pyramid
simpleONPyr, simpleOFFPyr = center_and_surround_feature(imgPyr=simplePyr, kernelSize=7, sigmaI=0.9, sigmaO=2.7)

# compute border cell on gray cell pyramid
BorderRefGray = border_ownership(refGrayPyr, refGrayONPyr, refGrayOFFPyr, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks)
dephasedBorderRefGray = border_ownership(refGrayPyr, refGrayONPyr, refGrayOFFPyr, standardGaborBanksPI, dephasedGaborBanksPI, dephasedVonMisesBanks, dephasedVonMisesBanksPI)
BorderCurGray = border_ownership(curGrayPyr, curGrayONPyr, curGrayOFFPyr, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks)
dephasedBorderCurGray = border_ownership(curGrayPyr, curGrayONPyr, curGrayOFFPyr, standardGaborBanksPI, dephasedGaborBanksPI, dephasedVonMisesBanks, dephasedVonMisesBanksPI)
# compute border cell on flicker cell pyramid
BorderFlicker = border_ownership(flickerPyr, flickerONPyr, flickerOFFPyr, standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks)
dephasedBorderFlicker = border_ownership(flickerPyr, flickerONPyr, flickerOFFPyr, standardGaborBanksPI, dephasedGaborBanksPI, dephasedVonMisesBanks, dephasedVonMisesBanksPI)
# compute border cell on color cell pyramid
BorderColor = []
dephasedBorderColor = []
for c in range(0, len(colorPyr)):
    BorderColor.append(border_ownership(colorPyr[c], colorONPyr[c], colorOFFPyr[c], standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks))
    dephasedBorderColor.append(border_ownership(colorPyr[c], colorONPyr[c], colorOFFPyr[c], standardGaborBanksPI, dephasedGaborBanksPI, dephasedVonMisesBanks, dephasedVonMisesBanksPI))
# compute border cell on gabor simple cell pyramid
BorderSimple = []
dephasedBorderSimple = []
for o in range(0, len(simplePyr)):
    BorderSimple.append(border_ownership(simplePyr[o], simpleONPyr[o], simpleOFFPyr[o], standardGaborBanks, dephasedGaborBanks, vonMisesBanks, dephasedVonMisesBanks))
    dephasedBorderSimple.append(border_ownership(simplePyr[o], simpleONPyr[o], simpleOFFPyr[o], standardGaborBanksPI, dephasedGaborBanksPI, dephasedVonMisesBanks, dephasedVonMisesBanksPI))

# grouping gray feature
groupCurGray = grouping_border_ownership(BorderCurGray, dephasedBorderCurGray, vonMisesBanks)
groupRefGray = grouping_border_ownership(BorderRefGray, dephasedBorderRefGray, vonMisesBanks)
# grouping flicker feature
groupFlicker = grouping_border_ownership(BorderFlicker, dephasedBorderFlicker, vonMisesBanks)
# grouping color features
groupColor = []
for o in range(0, len(BorderColor)):
    groupColor.append(grouping_border_ownership(BorderColor[o], dephasedBorderColor[o], vonMisesBanks))
# grouping simple cell features
groupSimple = []
for o in range(0, len(BorderSimple)):
    groupSimple.append(grouping_border_ownership(BorderSimple[o], dephasedBorderSimple[o], vonMisesBanks))

print("Gray grouping")
for l in range(0, len(groupCurGray)):
    plt.matshow(groupCurGray[l])
    plt.show()

print("Flicker grouping")
for l in range(0, len(groupFlicker)):
    plt.matshow(groupFlicker[l])
    plt.show()

print("Color grouping")
for c in range(0, len(groupColor)):
    for l in range(0, len(groupColor[0])):
        plt.matshow(groupColor[c][l])
        plt.show()

print("Edge grouping")
for c in range(0, len(groupSimple)):
    for l in range(0, len(groupSimple[0])):
        plt.matshow(groupSimple[c][l])
        plt.show()
