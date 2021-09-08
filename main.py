from sudoku_grid_detector import SudokuGridDetector

from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

def main():
    # Load and process a sudoku image
    t_start = time.time()
    sudoku_grid = SudokuGridDetector(img_filepath='inputs/full1.jpg')
    tot_time = time.time() - t_start

    print(f'Took {tot_time:.3f} seconds to load and process sudoku_grid image')

    # Generate intermediate images for plotting
    overlay = sudoku_grid.resized_img.copy()
    cv2.drawContours(overlay, [sudoku_grid.contour], -1, (00, 255, 0), 10)

    # Raw full-size image
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1), plt.imshow(sudoku_grid.raw_img, vmin=0, vmax=255)
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    # Binary image with green dots for corners
    plt.subplot(1, 4, 2), plt.imshow(sudoku_grid.binary_img, 'gray', vmin=0, vmax=255)
    plt.title('ADAPTIVE_THRESH_GAUSSIAN_C')
    plt.xticks([]), plt.yticks([])
    if sudoku_grid.corners is not None:
        plt.plot(sudoku_grid.corners[:, 0], sudoku_grid.corners[:, 1], 'go', markersize=15)

        # Overlaid resized image with sudoku_grid grid rectangle
        plt.subplot(1, 4, 3), plt.imshow(overlay, vmin=0, vmax=255)
        plt.title('OVERLAY')
        plt.xticks([]), plt.yticks([])

        # Warped, binarized and line-removed sudoku_grid grid
        plt.subplot(1, 4, 4), plt.imshow(sudoku_grid.sudoku_img, 'gray', vmin=0, vmax=255)
        plt.title('WARPED')
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()