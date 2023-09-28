import numpy as np
import cv2
import matplotlib.pyplot as plt


# Gaussian center and surround filter bank
def CenterAndSurroundFilterBank(Kernel_Size, Sigma_I, Sigma_O):
    CSON = None
    CSOFF = None
    # compute filter
    x, y = np.mgrid[:Kernel_Size, :Kernel_Size] - (Kernel_Size // 2)
    CSON = (1 / (2 * np.pi * Sigma_I**2)) * np.exp(-((x**2 + y**2) / (2 * Sigma_I**2))) - (
        1 / (2 * np.pi * Sigma_O**2)
    ) * np.exp(-((x**2 + y**2) / (2 * Sigma_O**2)))
    CSOFF = -(1 / (2 * np.pi * Sigma_I**2)) * np.exp(-((x**2 + y**2) / (2 * Sigma_I**2))) + (
        1 / (2 * np.pi * Sigma_O**2)
    ) * np.exp(-((x**2 + y**2) / (2 * Sigma_O**2)))
    return CSON, CSOFF


# Von Mises filter bank
def VonMisesFilterBank(Kernel_Size, RO, P, Orientation, Center):
    # init filter bank (single scale)
    VMB = []
    VMB_PI = []
    # create kernel grid
    for o in range(0, len(Orientation)):
        orient = np.deg2rad(Orientation[o] + 90)
        theta = orient
        theta_pi = orient + np.pi
        # Compute mesh grid
        x, y = np.mgrid[:Kernel_Size, :Kernel_Size] - (Kernel_Size // 2)
        x_pi, y_pi = np.mgrid[:Kernel_Size, :Kernel_Size] - (Kernel_Size // 2)
        # Compute angle and recenter if needed
        if Center:
            # compute offset
            x = x + RO * np.cos(theta)
            y = y + RO * np.sin(theta)
            x_pi = x_pi + RO * np.cos(theta_pi)
            y_pi = y_pi + RO * np.sin(theta_pi)
            # compute angle
            angle = np.arctan2(y, x)
            angle_pi = np.arctan2(y_pi, x_pi)
        else:
            # compute angle
            angle = np.arctan2(y, x)
            angle_pi = np.arctan2(y_pi, x_pi)
        # compute filter
        VMF = np.exp(P * RO * np.cos(angle - theta)) / np.i0(np.sqrt(x**2 + y**2) - RO)
        VMF_PI = np.exp(P * RO * np.cos(angle_pi - theta_pi)) / np.i0(np.sqrt(x_pi**2 + y_pi**2) - RO)
        # normalize
        VMF = VMF / np.max(np.max(VMF))
        VMF_PI = VMF_PI / np.max(np.max(VMF_PI))
        # append
        VMB.append(VMF)
        VMB_PI.append(VMF_PI)
    return VMB, VMB_PI


VMB, VMB_PI = VonMisesFilterBank(Kernel_Size=32, RO=10, P=0.2, Orientation=[0, 45, 90, 135], Center=True)
for i in range(len(VMB)):
    plt.matshow(VMB[i])
    plt.matshow(VMB_PI[i])
    plt.show()


# Border ownership computation at a single scale
def BorderOwnership(Feature_Map, On_Feature, Off_Feature, Von_Mises_Bank, Von_Mises_Bank_Pi):
    # Init mask
    B1 = []
    B2 = []
    # Iterate over orientation
    for o in range(len(Von_Mises_Bank)):
        #
    return None
