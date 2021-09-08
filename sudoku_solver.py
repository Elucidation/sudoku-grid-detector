import numpy as np
from abc import ABC, abstractmethod

class SudokuSolver():
  def __init__(self, max_iters=10000, max_depth=5):
    self.max_depth = max_depth
    self.max_iters = max_iters

  def solve(self, grid : np.ndarray) -> np.ndarray:
    """ Solved a given 9x9 2D array of integers.
    Returns a solved one. 
    Raises exception on failure """
    raise NotImplementedError()