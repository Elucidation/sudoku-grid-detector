import cv2
import numpy as np


class SudokuGridDetector:
    """Load Sudoku image, find grid and generate a process sudoku grid image"""

    def __init__(self, img_filepath : str, tilesize_px = 64):
        self.reset()
        self.img_filepath = img_filepath
        self.tilesize_px = tilesize_px
        self.process()

    def reset(self):
        self.img_filepath = None
        self.raw_img = None
        self.delta = None
        self.resized_img = None
        self.binary_img = None
        self.sudoku_img = None
        self.sudoku_cells = None
        self.tilesize_px = None
        self.contour = None
        self.corners = None
        self.M = None

    def process(self):
        """Load and process image into binarized sudoku grid image"""
        self.loadImage()
        if self.resized_img is None:
            raise ValueError('Resized image doesn\'t exist')
        self.binarizeImage()
        self.findSudokuGrid()
        if self.corners is not None:
            self.makeSudokuGridImage()
            self.generateSudokuCells()

    def loadImage(self):
        """Load raw image and generate resized image"""
        self.raw_img = cv2.imread(self.img_filepath)

        if self.raw_img is None:
            raise ValueError(f'Could not load {self.img_filepath}')

        # Convert to RGB for matplotlib
        self.raw_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)
        self.resizeImage()

    def resizeImage(self, max_height_px=1024):
        if self.raw_img is None:
            raise ValueError('Raw image not loaded yet')

        self.delta = max_height_px / self.raw_img.shape[0]
        print(f'Resizing {self.raw_img.shape} x {self.delta}')
        self.resized_img = cv2.resize(
            self.raw_img, None, fx=self.delta, fy=self.delta, interpolation=cv2.INTER_CUBIC
        )

    def binarizeImage(self):
        if self.resized_img is None:
            raise ValueError('Resized image doesn\'t exist')

        img_gray = cv2.cvtColor(self.resized_img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        self.binary_img = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)

    def findSudokuGrid(self, min_side_length_ratio=4):
        """Find the corners of the sudoku grid"""
        if self.binary_img is None:
            raise ValueError('Resized image doesn\'t exist')
        # Find enclosing rectangle
        contours = cv2.findContours(self.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        simple_contours = []
        # Look for a sudoku grid where the minimum pixel area is a ratio of smallest side of grid versus image shorter of width/height
        min_area = (min(self.binary_img.shape[0], self.binary_img.shape[1]) / min_side_length_ratio)**2
        best_contour = None
        for contour in contours:
            # If this contour area is > the current min area, replace best contour
            area = cv2.contourArea(contour)
            if area > min_area:
                epsilon = 0.1 * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
                # contour = cv2.convexHull(contour) # Orders points nicely
                if contour.shape[0] != 4:
                    continue
                best_contour = contour
                min_area = area

        if best_contour is None:
            print('Didn\'t find any contours that matched.')
            return
        
        self.contour = best_contour
        self.corners = self.orderPoints(np.float32(best_contour[:, 0, :]))

    def makeSudokuGridImage(self):
        """ Generate warped image """
        if self.corners is None:
            raise ValueError('Corners of image not found yet')
          # px
        output_pts = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * self.tilesize_px * 9
        # Get warp on input points for original image (un-scale with delta)
        self.M = cv2.getPerspectiveTransform(self.corners / self.delta, output_pts)
        raw_blurred_img = cv2.GaussianBlur(self.raw_img, (5, 5), 0) # Expensive, not really necessary
        warped = cv2.warpPerspective(
            raw_blurred_img, self.M, (self.tilesize_px * 9, self.tilesize_px * 9), flags=cv2.INTER_LINEAR
        )

        self.sudoku_img = cv2.adaptiveThreshold(
            cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            50,
        )

        # warped img size is width = height = self.tilesize_px * 9
        # remove pixels where lines are
        buffer_px = int(self.tilesize_px / 6) # Assume lines don't encroach > 1/6th of a tile
        for i in range(10):
            self.sudoku_img[:, max(0, i * self.tilesize_px - buffer_px) : min(self.tilesize_px * 9, i * self.tilesize_px + buffer_px)] = 0
            self.sudoku_img[max(0, i * self.tilesize_px - buffer_px) : min(self.tilesize_px * 9, i * self.tilesize_px + buffer_px), :] = 0

    def generateSudokuCells(self):
        """Generate 9x9xTxT cell array (T = tile size in pixels) from a 9x9 grid image, each tile is TxT pixels"""
        if self.sudoku_img.shape is None:
            raise ValueError('Sudoku image not generated yet')
        self.sudoku_cells = self.sudoku_img.reshape(9,self.tilesize_px,9,self.tilesize_px).swapaxes(1,2)

    def orderPoints(self, pts):
        """Re-order 4 corner points so top-left is first and clockwise."""
        out = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        out[0] = pts[np.argmin(s)]
        out[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        out[1] = pts[np.argmin(diff)]
        out[3] = pts[np.argmax(diff)]
        return out
