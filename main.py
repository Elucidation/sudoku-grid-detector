from sudoku_grid_detector import SudokuGridDetector
from sudoku_digit_recognizer import SudokuDigitRecognizer_CV

from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

def main():
    # Load and process a sudoku image
    t_start = time.time()
    tilesize_px = 64
    # sudoku_grid = SudokuGridDetector(img_filepath='inputs/full1.jpg', tilesize_px=tilesize_px)
    sudoku_grid = SudokuGridDetector(img_filepath='inputs/PXL_20210907_184648229.jpg', tilesize_px=tilesize_px)
    digit_recognizer = SudokuDigitRecognizer_CV('CV_digits', tilesize_px)
    prediction = digit_recognizer.recognizeDigits(sudoku_grid.sudoku_cells)
    tot_time = time.time() - t_start

    print('Sudoku Grid Prediction:')
    print(prediction)

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
    if sudoku_grid.corners is None:
        # Skip plotting the rest
        plt.tight_layout()
        plt.show()
        return

    plt.plot(sudoku_grid.corners[:, 0], sudoku_grid.corners[:, 1], 'go', markersize=15)

    # Overlaid resized image with sudoku_grid grid rectangle
    plt.subplot(1, 4, 3), plt.imshow(overlay, vmin=0, vmax=255)
    plt.title('OVERLAY')
    plt.xticks([]), plt.yticks([])

    # Warped, binarized and line-removed sudoku_grid grid
    plt.imsave(fname='suduko.png', arr=sudoku_grid.sudoku_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 4, 4), plt.imshow(sudoku_grid.sudoku_img, 'gray', vmin=0, vmax=255)
    for r in range(9):
        for c in range(9):
            if (prediction[r,c] > 0):
                plt.text(c*tilesize_px+tilesize_px*.8, r*tilesize_px+tilesize_px*.3, f'{prediction[r,c]}',c='yellow')
    plt.title('SUDOKU GRID')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    # plt.savefig('example1.jpg')
    plt.show()


if __name__ == '__main__':
    main()