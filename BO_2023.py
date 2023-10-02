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


# Border ownership computation at a single scale
def BorderOwnership(Feature_Map, ON_Filter_Bank, OFF_Filter_Bank, Von_Mises_Bank, Von_Mises_Bank_Pi, w):
    # Init mask
    B1 = []
    B2 = []
    # compute ON/OFF feature map
    on_feature_map = cv2.filter2D(Feature_Map, -1, ON_Filter_Bank)
    off_feature_map = cv2.filter2D(Feature_Map, -1, OFF_Filter_Bank)
    # Iterate over orientation
    for o in range(len(Von_Mises_Bank)):
        # ON + VM theta
        ON_theta = cv2.filter2D(on_feature_map, -1, Von_Mises_Bank[o])
        # OFF + VM theta pi
        OFF_theta_pi = cv2.filter2D(off_feature_map, -1, Von_Mises_Bank_Pi[o])
        # ON + VM theta pi
        ON_theta_pi = cv2.filter2D(on_feature_map, -1, Von_Mises_Bank_Pi[o])
        # OFF + VM theta
        OFF_theta = cv2.filter2D(off_feature_map, -1, Von_Mises_Bank[o])
        # compute first mask element
        M1 = np.maximum(0, (ON_theta - w * OFF_theta_pi)) + np.maximum(0, (OFF_theta - w * ON_theta_pi))
        # Compute second mask elment
        M2 = np.maximum(0, (ON_theta_pi - w * OFF_theta)) + np.maximum(0, (OFF_theta_pi - w * ON_theta))
        # Append border ownership
        B1.append(Feature_Map * M1)
        B2.append(Feature_Map * M2)
    return on_feature_map, off_feature_map, B1, B2


# Grouping procedure at a single scale
def Grouping(BorderOwnership_Map_1, BorderOwnership_Map_2, Von_Mises_Bank, Von_Mises_Bank_Pi):
    # Excitatory and Inhibitory grouping map init
    G1_EXC = cv2.filter2D(BorderOwnership_Map_1[0], -1, Von_Mises_Bank[0])
    G2_EXC = cv2.filter2D(BorderOwnership_Map_2[0], -1, Von_Mises_Bank_Pi[0])
    G1_INH = cv2.filter2D(BorderOwnership_Map_1[0], -1, Von_Mises_Bank_Pi[0])
    G2_INH = cv2.filter2D(BorderOwnership_Map_2[0], -1, Von_Mises_Bank[0])
    # Sum over orientation
    for o in range(1, len(Von_Mises_Bank)):
        G1_EXC = G1_EXC + cv2.filter2D(BorderOwnership_Map_1[o], -1, Von_Mises_Bank[o])
        G2_EXC = G2_EXC + cv2.filter2D(BorderOwnership_Map_2[o], -1, Von_Mises_Bank_Pi[o])
        G1_INH = G1_INH + cv2.filter2D(BorderOwnership_Map_1[o], -1, Von_Mises_Bank_Pi[o])
        G2_INH = G2_INH + cv2.filter2D(BorderOwnership_Map_2[o], -1, Von_Mises_Bank[o])
    # localize activation in inhibitory map
    G1_INH = np.absolute(G1_INH - np.max(G1_INH))
    G2_INH = np.absolute(G2_INH - np.max(G2_INH))
    # Final grouping
    group_map = (G1_EXC - G1_INH) + (G2_EXC - G2_INH)
    # return grouping map at single scale
    return group_map


img = cv2.imread("/home/visualbehavior/Desktop/kid_room.jpg", cv2.IMREAD_GRAYSCALE) / 255
ON_filter, OFF_filter = CenterAndSurroundFilterBank(Kernel_Size=32, Sigma_I=0.90, Sigma_O=2.70)
VM, VM_pi = VonMisesFilterBank(Kernel_Size=32, RO=10, P=0.2, Orientation=[0, 45, 90, 135], Center=True)

plt.matshow(img)
plt.show()

on_map, off_map, border_one, border_two = BorderOwnership(
    Feature_Map=img,
    ON_Filter_Bank=ON_filter,
    OFF_Filter_Bank=OFF_filter,
    Von_Mises_Bank=VM,
    Von_Mises_Bank_Pi=VM_pi,
    w=3,
)

plt.matshow(on_map)
plt.matshow(off_map)
plt.show()

for i in range(len(border_one)):
    plt.matshow(border_one[i])
    plt.matshow(border_two[i])
    plt.show()

grouping_map = Grouping(
    BorderOwnership_Map_1=border_one, BorderOwnership_Map_2=border_two, Von_Mises_Bank=VM, Von_Mises_Bank_Pi=VM_pi
)

plt.matshow(grouping_map)
plt.show()
