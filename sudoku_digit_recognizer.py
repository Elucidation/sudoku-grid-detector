import numpy as np
from abc import ABC, abstractmethod

class SudokuDigitRecognizer(ABC):
  @abstractmethod
  def recognizeDigits(self, grid : np.ndarray) -> int:
    """ Given 9x9xTxT array of sudoku tiles, identify each. """
    pass

  @abstractmethod
  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    pass


class SudokuDigitRecognizer_CV(SudokuDigitRecognizer):
  """CV Template matching-based digit recognizer"""
  def __init__(self, template_path : str, tilesize_px : int = 64):
    self.template_path = template_path
    self.tilesize_px = tilesize_px
    self.loadDigits()

  def loadDigits(self):
    self.digits = np.zeros([10,self.tilesize_px,self.tilesize_px], dtype=int)
    for i in range(10):
      digit_path = f'{self.template_path}/digit_{i}.txt'
      self.digits[i,:,:] = np.loadtxt(digit_path, dtype=int)

    # Convert to correlation matrix of -1 and 1 values
    # self.digits = self.digits * ( 2. / 255.0 ) - 1.0

  def recognizeDigits(self, grid : np.ndarray) -> int:
    """ Given 9x9xTxT array of sudoku tiles, identify each. """
    grid_prediction = np.zeros([9,9],dtype=int)
    for r in range(9):
      for c in range(9):
        grid_prediction[r,c] = self.recognizeDigit(grid[r,c,:,:])
    return grid_prediction

  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    probabilities = np.zeros(10, dtype = np.float)
    # TODO : Center digits in tile first.
    for i in range(10):
      probabilities[i] = np.sum(self.digits[i,:,:] * tile) # Correlation-based
    return np.argmax(probabilities)


class SudokuDigitRecognizer_ML(SudokuDigitRecognizer):
  """CV Template matching-based digit recognizer"""
  def __init__(self, model):
    self.model = model

  def recognizeDigits(self, grid : np.ndarray) -> int:
    """ Given 9x9xTxT array of sudoku tiles, identify each. """
    raise NotImplementedError()

  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    raise NotImplementedError()