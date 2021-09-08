import numpy as np
from abc import ABC, abstractmethod

class SudokuDigitRecognizer(ABC):
  @abstractmethod
  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    pass


class SudokuDigitRecognizer_CV(SudokuDigitRecognizer):
  """CV Template matching-based digit recognizer"""
  def __init__(self, template_path):
    self.template_path = template_path

  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    raise NotImplementedError()

    
class SudokuDigitRecognizer_ML(SudokuDigitRecognizer):
  """CV Template matching-based digit recognizer"""
  def __init__(self, model):
    self.model = model

  def recognizeDigit(self, tile : np.ndarray) -> int:
    """ Detects the digit for a given binary image 64x64 tile.
    Returns the predicted integer digit. """
    raise NotImplementedError()
