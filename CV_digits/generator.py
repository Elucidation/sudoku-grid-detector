# Script to generate CV digits
from matplotlib import pyplot as plt
import numpy as np
import cv2

def manualGenDigits(sudoku_cells):
    # 0 - 9 digits for full1.jpg
    digits_r = [0,1,5,6,7,8,8,0,5,6]
    digits_c = [0,8,3,0,0,3,0,5,1,4]
    digits = sudoku_cells[digits_r, digits_c, : ,:]
    return digits

data = cv2.imread('sudoku.png',0)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((3,3), np.uint8)
# ksize
ksize = (5, 5)
  
# Using cv2.blur() method 
# data = cv2.blur(data, ksize)
data = cv2.GaussianBlur(data,ksize,cv2.BORDER_DEFAULT)
data = np.uint8(data * (255.0 / data.max())) # push back to 1.0
# data = cv2.dilate(data, kernel, iterations=1)

plt.imshow(data,cmap='gray',vmin=0, vmax=255)
plt.show()

tilesize_px = 64
sudoku_cells = data.reshape(9,tilesize_px,9,tilesize_px).swapaxes(1,2)
digits = manualGenDigits(sudoku_cells)
for i in range(10):
  np.savetxt(f'digit_{i}.txt', digits[i,:,:], fmt='%d')