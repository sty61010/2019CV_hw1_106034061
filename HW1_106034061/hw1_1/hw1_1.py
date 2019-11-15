#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:22:24 2019

@author: cengjingyu
"""

import os
import cv2 
import math
import random
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import data, io, filters, color
from scipy import ndimage as ndi
from PIL import Image


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
        
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    #print(kernel_2D)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

def convolution(image, kernel, average=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row ) / 2)### to INT
    pad_width = int((kernel_col ) / 2)### to INT
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

def gaussian_smooth(img, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=5)
    kernel/=kernel.sum()
    blur_img=cv2.filter2D(img,-1,kernel)
    return blur_img

def sobel_edge_detection(img):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img = color.rgb2gray(img)
    gx = convolution(img, filter)
    gy = convolution(img, np.flip(filter.T, axis=0))
 
    mag_img = np.sqrt(np.square(gx) + np.square(gy))
    mag_img = np.floor(mag_img * 255.0 / np.max(mag_img))

    mag = np.sqrt(gx ** 2 + gy ** 2)
    h, w = img.shape
    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hsv[..., 1] = np.ones((h, w)) # or just write = 1.0
    hsv[..., 2] = (mag - mag.min()) / (mag.max() - mag.min())
    dir_img = color.hsv2rgb(hsv)
    return [mag_img, dir_img, gx, gy] 
'''
def sobel_edge_detection(img):
    #image = color.rgb2gray(image)
    avg = np.array([1, 2, 1])
    diff = np.array([1, 0, -1])

    m, n = img.shape[:2]
    gx = np.zeros(img.shape)
    gy = np.zeros(img.shape)

    for i in range(m):
        gx[i,:] = np.convolve(img[i,:], diff, mode='same')
        gy[i,:] = np.convolve(img[i,:], avg, mode='same')

    for i in range(n):
        gx[:,i] = np.convolve(gx[:,i], avg, mode='same')
        gy[:,i] = np.convolve(gy[:,i], diff, mode='same')

    mag_img = np.sqrt(gx**2 + gy**2)
    mag_img = np.floor(mag_img * 255.0 / np.max(mag_img))

    #gx += np.ones(gx.shape) * 0.00001
    #dir_img = np.arctan(gy / gx)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    h, w = img.shape
    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hsv[..., 1] = np.ones((h, w)) # or just write = 1.0
    hsv[..., 2] = (mag - mag.min()) / (mag.max() - mag.min())
    dir_img = color.hsv2rgb(hsv)
    return [mag_img, dir_img, gx, gy]
'''
def structure_tensor(dx, dy, size):
    Axx = gaussian_smooth(dx * dx, size)
    Axy = gaussian_smooth(dx * dy, size)
    Ayy = gaussian_smooth(dy * dy, size)
    
    det = Axx * Ayy - Axy * Axy
    tr = Axx + Axy
    R = det - 0.04 * tr * tr
    return R

def nms(img, R, size):
    mask1 = (R > 5e-6)
    mask2 = (np.abs(ndi.maximum_filter(R, size) - R) < 1e-14)
    mask = (mask1 & mask2)

    result=img.copy()
    result[mask>0.001*mask.max()] = [0, 0, 255]
    r, c = np.nonzero(mask)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(img, cmap='viridis')
    ax.plot(c, r, 'r.', markersize=3)
    
    return result
    
if __name__ == '__main__':
    img_num = 2
    img_folder_path = 'images'
    img_folder_save_path = 'results'
    
    for idx in range(img_num):
        img_path = os.path.join(img_folder_path, str(idx+1)+'.jpg')
        print(img_path)
        img = cv2.imread(img_path) 
        
        #################################################################
        #Gaussion Smooth#################################################
        #Kernel=5, sigma=5###############################################
        print("->>>>>Kernel size=5, Sigma=5")
        print("Gaussian Smoothing.....")
        img_GS_5 = gaussian_smooth(img, 5)
        #### Save Results
        save_name = str(idx+1)+'_GS_5'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_GS_5)
        
        #Kernel=10, sigma=5###############################################
        print("->>>>>Kernel size=10, Sigma=5")
        print("Gaussian Smoothing.....")
        img_GS_10=gaussian_smooth(img, 10)
        #### Save Results
        save_name = str(idx+1)+'_GS_10'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_GS_10)
        
        #Sobel Edge Detection############################################
        print("Sobel Edge Detection.....")
        img_SED1_5, img_SED2_5, gx, gy=sobel_edge_detection(img_GS_5)

        #### Save Results
        save_name = str(idx+1)+'_SED_M_5'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_SED1_5)
        
        plt.figure()
        plt.imshow(img_SED2_5, cmap='viridis')
        save_name = str(idx+1)+'_SED_D_5'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        plt.savefig(save_img_path)
        
        #Sobel Edge Detection############################################
        print("Sobel Edge Detection.....")
        img_SED1_10, img_SED2_10, gx, gy=sobel_edge_detection(img_GS_10)

        #### Save Results
        save_name = str(idx+1)+'_SED_M_10'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_SED1_10)
        
        plt.figure()
        plt.imshow(img_SED2_10, cmap='viridis')
        save_name = str(idx+1)+'_SED_D_10'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        plt.savefig(save_img_path)

        #Structure Tensor################################################
        print("Structure Tensor.....")
        img_ST_3=structure_tensor(gx, gy, 3)
        
        print("Structure Tensor.....")
        img_ST_30=structure_tensor(gx, gy, 30)
        #NMS ############################################################
        print("NMS.....")
        img_NMS_3=nms(img, img_ST_3, 30)
        #### Save Results
        save_name = str(idx+1)+'_NMS_3'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_NMS_3)
        print("NMS.....")
        img_NMS_30=nms(img, img_ST_30, 30)
        #### Save Results
        save_name = str(idx+1)+'_NMS_30'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_NMS_30)
        
        #################################################################
        #Rotate##########################################################
        img = Image.open(img_path)
        img = img.rotate(30)
        img = np.asarray(img)
        #Gaussion Smooth#################################################        
        #Kernel=10, sigma=5###############################################
        print("->>>>>Rotated")
        print("Gaussian Smoothing.....")
        img_GS_10=gaussian_smooth(img, 10)
        
        #Sobel Edge Detection############################################
        print("Sobel Edge Detection.....")
        img_SED1_10, img_SED2_10, gx, gy=sobel_edge_detection(img_GS_10)

        #Structure Tensor################################################
        print("Structure Tensor.....")
        img_ST_10=structure_tensor(gx, gy, 10)
            
        #NMS ############################################################
        print("NMS.....")
        img_NMS_RT=nms(img, img_ST_10, 30)
        #### Save Results
        save_name = str(idx+1)+'_NMS_RT'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_NMS_RT)
        
        #################################################################
        #Resize##########################################################
        img = cv2.imread(img_path)
        h, w, c=img.shape
        img = cv2.resize(img,None, fx=0.5, fy=0.5)
        #Gaussion Smooth#################################################        
        #Kernel=10, sigma=5###############################################
        print("->>>>>Resized")
        print("Gaussian Smoothing.....")
        img_GS_10=gaussian_smooth(img, 10)
        
        #Sobel Edge Detection############################################
        print("Sobel Edge Detection.....")
        img_SED1_10, img_SED2_10, gx, gy=sobel_edge_detection(img_GS_10)

        #Structure Tensor################################################
        print("Structure Tensor.....")
        img_ST_10=structure_tensor(gx, gy, 10)
            
        #NMS ############################################################
        print("NMS.....")
        img_NMS_RS=nms(img, img_ST_10, 30)
        #### Save Results
        save_name = str(idx+1)+'_NMS_RS'+'.png'
        save_img_path = os.path.join(img_folder_save_path, save_name)
        cv2.imwrite(save_img_path, img_NMS_RS)
    print('') # for next image
