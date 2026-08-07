import torch
import torch.nn.functional as F
from torch.nn.functional import pad
from utils.utils_image import *
from utils.utils_deblur import *
import numpy as np
import math
import cv2 as cv

def padPSF(PSF, m, n=None):
    """
    Pad a PSF tensor with zeros to make it bigger.

    Args:
        PSF: Tensor containing the point spread function.
        m: Desired row dimension of the padded tensor.
        n: Desired column dimension of the padded tensor (optional).

    Returns:
        P: Padded m-by-n tensor.
    """
    # Set default parameters
    if n is None:
        if isinstance(m, int):
            n = m
        else:
            n = m[1]
            m = m[0]
    # Pad the PSF with zeros
    P = torch.zeros((m, n))
    P[:PSF.shape[0], :PSF.shape[1]] = PSF
    return P   

def genaratePsf(length,angle):
    EPS=np.finfo(float).eps                                 
    alpha = (angle-math.floor(angle/ 180) *180) /180* math.pi
    cosalpha = math.cos(alpha)  
    sinalpha = math.sin(alpha)  
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:  
        xsign = 1
    psfwdt = 1;  
    #模糊核大小
    sx = int(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))  
    sy = int(math.fabs(length*sinalpha + psfwdt - length*EPS))
    psf1=np.zeros((sy,sx))
#psf1是左上角的权值较大，越往右下角权值越小的核。
    #这时运动像是从右下角到左上角移动
    for i in range(0,sy):
        for j in range(0,sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j) 
            # if  rad >= half and math.fabs(psf1[i][j]) <= psfwdt:  
            #     temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)  
            #     psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);  
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    #运动方向是往左上运动，锚点在（0，0）
    anchor=(0,0)
    #运动方向是往右上角移动，锚点一个在右上角
    #同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle<90 and angle>0:
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,0)
    elif angle>-90 and angle<0:#同理：往右下角移动
        psf1=np.flipud(psf1)
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
    elif anchor<-90:#同理：往左下角移动
        psf1=np.flipud(psf1)
        anchor=(0,psf1.shape[0]-1)
    psf1=psf1/psf1.sum()
    return psf1,anchor