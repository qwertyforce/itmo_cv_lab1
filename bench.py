import cv2
from timeit import default_timer as timer
import numpy as np
import statistics
from tqdm import tqdm
from numba import jit

img1 = cv2.imread("./1.jpg",0)
kernel_sizes = [i for i in range(64) if i % 2 == 1]
print(kernel_sizes)
res = []
res2 = []
res3 = []

def median_filter(img, kernel_size):
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            med = np.uint8(np.median(img[y:y+kernel_size,x:x+kernel_size].flatten()))
            img[y][x]=med
    return img

@jit(nopython=True,fastmath=True)
def numba_median_filter(img, kernel_size):
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            med = np.uint8(np.median(img[y:y+kernel_size,x:x+kernel_size].flatten()))
            img[y][x]=med
    return img

for _ in range(5): 
    numba_median_filter(img1.copy(),3)

for ks in tqdm(kernel_sizes):
    temp_res = []
    temp_res2 = []
    temp_res3 = []

    for _ in range(100):
        start = timer()
        cv2.medianBlur(img1.copy(),ks)
        end = timer()
        elapsed_time = 1000*(end - start) #ms
        temp_res.append(elapsed_time)
    res.append((ks,statistics.mean(temp_res)))

    for _ in range(100):
        start = timer()
        median_filter(img1.copy(),ks)
        end = timer()
        elapsed_time = 1000*(end - start) #ms
        temp_res2.append(elapsed_time)
    res2.append((ks,statistics.mean(temp_res2)))

    for _ in range(100):
        start = timer()
        numba_median_filter(img1.copy(),ks)
        end = timer()
        elapsed_time = 1000*(end - start) #ms
        temp_res3.append(elapsed_time)
    res3.append((ks,statistics.mean(temp_res3)))

print(res)
print(res2)
print(res3)