# Script to generate CV digits
from matplotlib import pyplot as plt
import numpy as np
import cv2

def manualGenDigits(sudoku_cells):
    # 0 - 9 digits for full1.jpg
    digits_r = [0,1,5,6,7,8,8,0,5,6]
    digits_c = [0,8,3,0,0,3,0,5,1,4]
    digits = sudoku_cells[digits_r, digits_c, : ,:]
    # Center each digit
    for i in range(1,10):
      digits[i,:,:] = getCenteredTile(digits[i,:,:])
    return digits

def getCentroid(tile: np.ndarray) -> [np.float, np.float]:
  x = range(0, tile.shape[0])
  y = range(0, tile.shape[1])
  (X,Y) = np.meshgrid(x,y)

  tile_sum = tile.sum()
  if tile_sum == 0:
    # no data, return current center
    return [tile.shape[0] / 2.0, tile.shape[1] / 2.0]
   
  cx = np.float((X*tile).sum()) / tile_sum
  cy = np.float((Y*tile).sum()) / tile_sum
  return [cx,cy]

def getCenteredTile(tile: np.ndarray) -> np.ndarray:
  # Given TxT tile np array, center centroid and return
  centroid = np.array(getCentroid(tile), dtype=np.float)
  offset = np.array(tile.shape) / 2.0 - centroid
  T = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
  centered_tile = cv2.warpAffine(tile, T, tile.shape)
  return centered_tile

def main():
  data = cv2.imread('sudoku.png',0)
  data = cv2.dilate(data, (3, 3), iterations=1)
  data = cv2.GaussianBlur(data,(3, 3),cv2.BORDER_DEFAULT)
  data = np.uint8(data * (255.0 / data.max())) # push back to 1.0

  # plt.imshow(data,cmap='gray',vmin=0, vmax=255)
  # plt.show()

  tilesize_px = 64
  sudoku_cells = data.reshape(9,tilesize_px,9,tilesize_px).swapaxes(1,2)
  digits = manualGenDigits(sudoku_cells)

  for i in range(10):
    np.savetxt(f'digit_{i}.txt', digits[i,:,:], fmt='%d')
  plt.figure(figsize=(10,10))
  for r in range(10):
    plt.subplot(3,4,r+1); plt.imshow(digits[r,:,:], 'gray')
    plt.grid()
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()