import cv2
import numpy as np
import math
import argparse
from os.path import join, basename
from tqdm import tqdm
from glob import glob
from numpy import *

# 1, Brenner function
def Brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

# 2, Laplacian function
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()

# 3, SMD（灰度方差）
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(1, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

# 4, SMD2（灰度方差乘积）
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

# 5, 方差函数
def Variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

# 6, Energy functions
def Energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out

# 7, Vollath functions
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

# 8, 熵函数
def Entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True, type=str, help='folder containing images to test metrics')
    args = parser.parse_args()

    brenner_metrics = []
    laplacian_metrics = []
    smd_metrics = []
    smd2_metrics = []
    variance_metrics = []
    energy_metrics = []
    vollath_metrics = []
    entropy_metrics = []
    
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png'}
    
    image_folder_list = glob(join(args.img_folder, '*'))
    for image_path in tqdm(image_folder_list):
        if basename(image_path).split('.')[-1] in IMAGE_EXTENSIONS:
            image = cv2.imread(image_path)
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            brenner = Brenner(img_gray)
            laplacian = Laplacian(img_gray)
            smd = SMD(img_gray)
            smd2 = SMD2(img_gray)
            variance = Variance(img_gray)
            energy = Energy(img_gray)
            vollath = Vollath(img_gray)
            entropy = Entropy(img_gray)

            brenner_metrics.append(brenner)
            laplacian_metrics.append(laplacian)
            smd_metrics.append(smd)
            smd2_metrics.append(smd2)
            variance_metrics.append(variance)
            energy_metrics.append(energy)
            vollath_metrics.append(vollath)
            entropy_metrics.append(entropy)
        else:
            print('Warning: unknown Image Format.')
    
    print(" brenner: ", mean(brenner_metrics))
    print(" laplacian: ", mean(laplacian_metrics))
    print(" smd: ", mean(smd_metrics))
    print(" smd2: ", mean(smd2_metrics))
    print(" variance: ", mean(variance_metrics))
    print(" energy: ", mean(energy_metrics))
    print(" vollath: ", mean(vollath_metrics))
    print(" entropy: ", mean(entropy_metrics))