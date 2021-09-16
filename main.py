from sudoku_grid_detector import SudokuGridDetector
from sudoku_digit_recognizer import SudokuDigitRecognizer_CV

from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import time

def processImage(img_filepath : str, tilesize_px : int) -> np.ndarray:
    t_start = time.time()
    sudoku_grid = SudokuGridDetector(img_filepath=img_filepath, tilesize_px=tilesize_px)
    t_end = time.time() - t_start
    print(f'Took {t_end:.3f} seconds to load image and find the sudoku grid')

    t_start = time.time()
    digit_recognizer = SudokuDigitRecognizer_CV('CV_digits', tilesize_px)
    prediction = digit_recognizer.recognizeDigits(sudoku_grid.sudoku_cells)
    t_end = time.time() - t_start
    print(f'Took {t_end:.3f} seconds to detect the digits in the sudoku grid')
    return sudoku_grid, prediction

def prettyPrintGrid(grid : np.ndarray):
    """ Given 9x9 sudoku grid, return pretty string. """
    msg = ''
    grid = grid.astype('str')
    for i in range(9):
        msg += ''.join(grid[i,0:3]) + '|' + ''.join(grid[i,3:6]) + '|' + ''.join(grid[i,6:9]) + '\n'
        if i in [2,5]:
            msg += '---+---+---\n'
    msg = msg.replace('0',' ')
    return msg

def plotResults(sudoku_grid : SudokuGridDetector, prediction: np.ndarray, tilesize_px : int):
    # Generate intermediate images for plotting
    overlay = sudoku_grid.resized_img.copy()
    cv2.drawContours(overlay, [sudoku_grid.contour], -1, (00, 255, 0), 10)

    # Raw full-size image
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1), plt.imshow(sudoku_grid.raw_img, vmin=0, vmax=255)
    plt.title('Original Image')
    # plt.xticks([]), plt.yticks([])

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
    msg = prettyPrintGrid(prediction)
    plt.text(10,10,msg,fontname='Consolas',horizontalalignment='left',verticalalignment='top', fontsize=12, color='yellow')

    # Warped, binarized and line-removed sudoku_grid grid
    plt.imsave(fname='suduko.png', arr=sudoku_grid.sudoku_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 4, 4), plt.imshow(sudoku_grid.sudoku_img, 'gray', vmin=0, vmax=255)
    for r in range(9):
        for c in range(9):
            if (prediction[r,c] > 0):
                plt.text(c*tilesize_px+tilesize_px*.8, r*tilesize_px+tilesize_px*.3, f'{prediction[r,c]}',c='yellow')
    plt.title('SUDOKU GRID (Yellow is predicted digit)')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

def main():
    # Load and process a sudoku image
    tilesize_px = 64 # hard-coded for CV_digits for now.

    # img_filepath='inputs/full1.jpg'
    for i, img_filepath in enumerate(glob.glob('inputs/*.jpg')):

        # Time processing of image into predicted sudoku grid and solving it
        t_start = time.time()
        sudoku_grid, prediction = processImage(img_filepath, tilesize_px)
        tot_time = time.time() - t_start
        print(f'Took {tot_time:.3f} seconds total to process an input image')

        print('Sudoku Grid Prediction:')
        print(prediction)

        # Visualize 
        plotResults(sudoku_grid, prediction, tilesize_px)
        plt.suptitle(img_filepath)
        plt.savefig(f'results/result_{i:02d}.jpg')
    # plt.show()


if __name__ == '__main__':
    main()