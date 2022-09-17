import cv2
import numpy as np
from time import sleep
import keyboard
from numba import jit
from tqdm import tqdm

img1 = cv2.imread("./3.jpg")
kernel_size = 7

def median_filter(img, kernel_size):
    edge = kernel_size//2
    new_img=np.zeros((img.shape[0]-2*edge,img.shape[1]-2*edge),dtype=np.uint8)
    for y in tqdm(range(edge,img.shape[0]-edge)):
        for x in range(edge,img.shape[1]-edge):
            new_img[y-edge][x-edge]= np.median(img[y-edge:y+kernel_size-edge,x-edge:x+kernel_size-edge])
    return new_img

@jit(nopython=True,fastmath=True)
def numba_median_filter(img, kernel_size):
    edge = kernel_size//2
    new_img=np.zeros((img.shape[0]-2*edge,img.shape[1]-2*edge),dtype=np.uint8)
    for y in range(edge,img.shape[0]-edge):
        for x in range(edge,img.shape[1]-edge):
            new_img[y-edge][x-edge]= np.median(img[y-edge:y+kernel_size-edge,x-edge:x+kernel_size-edge])
    return new_img

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.uint8(gray)

print("ready") 
while True:
    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
       # cv2.imshow('img',cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY))

        rgb_img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        cv2.imshow('img',rgb2gray(rgb_img))

        cv2.waitKey(0)
        print('You Pressed A Key!')
        sleep(0.2)

    if keyboard.is_pressed('w'):  # if key 'q' is pressed 
        gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # _img = cv2.medianBlur(gray_img,kernel_size)
        edge = kernel_size//2
        _img = cv2.copyMakeBorder(gray_img, edge, edge, edge, edge, cv2.BORDER_REPLICATE)
        # _img = median_filter(_img.copy(),kernel_size)

        _img = numba_median_filter(_img.copy(),kernel_size)

        # cv2.imwrite("med_blur_test8.jpg",_img)
        _img = np.uint8(_img)
        print(_img.shape)
        cv2.imshow('img',_img)
        cv2.waitKey(0)
        print('You Pressed A Key!')
        sleep(0.2)





# def python_rgb_med_blur(img,kernel_size):
#     img_b, img_g, img_r = cv2.split(img)
#     _img_b = np.uint8(median_filter(img_b, kernel_size))
#     _img_g = np.uint8(median_filter(img_g, kernel_size))
#     _img_r = np.uint8(median_filter(img_r, kernel_size))
#     merged = cv2.merge([_img_b, _img_g, _img_r])
#     return merged