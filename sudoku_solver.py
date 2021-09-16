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
    solution = self.bfs_sudoku_solve(grid)
    if solution is None:
      raise ValueError('Unable to find solution')
    return solution.grid

  def bfs_sudoku_solve(self, grid : np.ndarray, counter_limit : int =100000) -> np.ndarray:
    start_time = time.time()
    last = start_time

    p_start = PartialSudoku(grid) 
    p_start.resolveAnalytically()
    if not p_start.isLegal():
      print(f"Given sudoku is not legal : {p_start}")
      return False
    if p_start.isComplete():
      print(f"Found solution analytically")
      return p_start

    q = deque()
    q.append(p_start)

    counter = 0
    latest_best_score = 0
    latest_best = None
    while True:
      if counter > counter_limit:
        print(f"Hit iteration limit after checking {counter} solutions")
        print(f"Best: {latest_best}")
        break
      curr_pset = q.popleft()
      new_partial_sudokus = curr_pset.genPotentialSudokus(9 * 9)
      for new_pset in new_partial_sudokus:
        counter += 1
        new_pset.resolveAnalytically()
        if not new_pset.isLegal():
          continue
        if new_pset.getCompleteness() > latest_best_score:
          latest_best_score = new_pset.getCompleteness()
          latest_best = new_pset
          print(f"Counter: {counter} | queue size {len(q)}")
          print(latest_best)
        if new_pset.isComplete():
          print(f"Found solution after {counter} partials")
          return new_pset
        q.append(new_pset)
      if time.time() - last > 5:
        last = time.time()
        print(
          f"T: {time.time() - start_time:.2f} Counter: {counter} | queue size {len(q)}"
        )
        print(curr_pset)
    return None


"""
Goal: given 9x9 digits (0 = empty), solve a classic sudoku

Theory:
A - brute force try possible combinations until a solution is found
B - use human strategies (naked singles, etc.) to solve
C - use combination of B & A

Optimizations:
Keep a dict of empty cells and their potential values: dict[(row,col)] = set(potential digits)
Each potential solution is a full mapping of those empty cells to digits: set((row,col,value))
Each partial solution is a partial mapping of those empty cells to digits: set((row,col,value))
*No need to use sets of course, just ignoring ordering here.

A potential solution in the worst case is an empty sudoku 9x9 with 9 possibilities each
ie. 729 bits to store a single potential solution, and 81**9 possible solutions, but less valid ones.
A possibly space efficient method to store a solution is 729 bits instead of a dict of sets etc, however there's
wasted space for those cells that already have a value in them.
This optimization can probably be tested later.
Simple fixed-size way to store partial potential solution is a 9x9x9 grid, the 3rd axis is possible values
Nice thing there is removing possibliites is matrix ops

A full sudoku validation could be run every time, or only each time we add a digit.


Concerns:
A recursive solution has a stack depth of 9x9=81 max, but due to # of possible solutions probably can't use.
Non-recursive may also use up too much stack memory

TODO: A*, weight priority queue based on completeness? shortest possibility count?

"""

import numpy as np
import copy
from collections import defaultdict
from collections import OrderedDict
from collections import deque
import queue  # PriorityQueue
import time


class PartialSudoku:
  """Partial Sudoku state"""

  def __init__(self, grid, possibles=None, legal=None, depth=0):
    self.grid = grid.copy()
    # todo don't use deepcopy here
    if possibles is None:
      self.possibles, self.min_length = self.getGridLegalPossibilities()
    else:
      self.possibles = copy.deepcopy(possibles)
    if legal is None:
      self.legal = self.checkLegal()
    else:
      self.legal = legal
    self.complete = not 0 in self.grid
    self.depth = depth

  def resolveAnalytically(self):
    # Attempt to use common sudoku techniques to resolve cells
    # until we hit a dead end for brute forcing
    # Keep going as long as cells change
    counter = 0
    while self.legal and not self.complete:
      counter += 1
      if counter > 1000:
        raise Exception("TODO: Safety check: Hit an infinite loop")
        return
      if self.resolveNakedSingle():
        continue
      if self.resolveSingleInBox():
        # With these two, s16.txt takes ~17 seconds
        continue
      if self.resolveInlineSingle():
        # With these three, s16.txt takes ~5 seconds
        continue
      # No changes, reached a dead end
      return

  def resolveNakedSingle(self):
    # If a naked single exists, add it in
    if self.legal and not self.complete and self.min_length == 1:
      (r, c) = next(iter(self.possibles))
      vals = self.possibles[(r, c)]
      val = next(iter(vals))
      # print('Resolving naked single')
      self.addValue(r, c, val)
      return True
    return False

  def resolveSingleInBox(self):
    # Look in each 3x3 box to resolve a value that can only exist in one cell
    # TODO : remove multi-for-loops
    for box_r in [0, 3, 6]:
      for box_c in [0, 3, 6]:
        # count all the possible values in the box, if a digit only shows once, then
        # it must be filled in that spot
        counts = np.zeros(9, dtype=int)
        for i in range(3):
          for j in range(3):
            if (box_r + i, box_c + j) in self.possibles:
              for v in self.possibles[(box_r + i, box_c + j)]:
                counts[v - 1] += 1
        for k in range(9):
          if counts[k] == 1:
            digit = k + 1
            # The digit is only possible in one cell of the 3x3
            for i in range(3):
              for j in range(3):
                if (box_r + i, box_c + j) in self.possibles:
                  if digit in self.possibles[(box_r + i, box_c + j)]:
                    # print('Resolving single in box')
                    self.addValue(box_r + i, box_c + j, digit)
                    return True
    return False

  def resolveInlineSingle(self):
    # Todo, look in boxes where possibilities are only
    # Along 1 row/col, and use to reduce possibilities in other rows
    # For each row and column, count possible values,
    # if ever 1, that must be where the value goes.
    for r in range(9):
      # Row
      counts = np.zeros(9, dtype=int)
      for i in range(9):
        pos = (r, i)
        if pos in self.possibles:
          for v in self.possibles[pos]:
            counts[v - 1] += 1
      for k in range(9):
        if counts[k] == 1:
          # The digit is only possible in one cell of this row
          digit = k + 1
          for i in range(9):
            pos = (r, i)
            if pos in self.possibles:
              if digit in self.possibles[pos]:
                # print('Resolving single in row')
                self.addValue(pos[0], pos[1], digit)
                return True

      for c in range(9):
        # Col
        counts = np.zeros(9, dtype=int)
        for i in range(9):
          pos = (i, c)
          if pos in self.possibles:
            for v in self.possibles[pos]:
              counts[v - 1] += 1
        for k in range(9):
          if counts[k] == 1:
            # The digit is only possible in one cell of this row
            digit = k + 1
            for i in range(9):
              pos = (i, c)
              if pos in self.possibles:
                if digit in self.possibles[pos]:
                  # print('Resolving single in column')
                  self.addValue(pos[0], pos[1], digit)
                  return True

    return False

  def getPriority(self):
    (r, c) = next(iter(self.possibles))
    vals = self.possibles[(r, c)]
    return len(vals)

  def addValue(self, r, c, val):
    # print(f'Adding {val} to {r} x {c}')
    if not self.legal:
      return
    self.grid[r, c] = val
    if val not in self.possibles[(r, c)]:
      self.legal = False
      return
    self.complete = not 0 in self.grid
    # Todo, check only affected row/col/box
    if self.complete:
      self.possibles = None
      self.min_length = None
      return
    self.possibles, self.min_length = self.getGridLegalPossibilities()
    if not self.complete and self.min_length == 0:
      self.legal = False

  def isLegal(self):
    return self.legal

  def isComplete(self):
    return self.complete

  def getCompleteness(self):
    # Returns number of digits filled so far (up to 9x9 = 81)
    return float(81 - sum(self.grid.flatten() == 0))

  def genchoices(self):
    # Return dict (r,c)->[choice,choice...] values sorted by length of possible choices
    # ie. first ones are cells with fewest choices
    choices = []
    for (r, c), vals in self.possibles.items():
      for val in vals:
        yield (r, c, val)

  def genPotentialSudokus(self, n):
    pot_sudokus = []
    if not self.legal:
      return pot_sudokus
    i = 0
    for choice in self.genchoices():
      new_p = PartialSudoku(self.grid, self.possibles, self.legal, self.depth + 1)
      new_p.addValue(*choice)
      # print(i, choice, new_p.grid, new_p.isLegal())

      if new_p.isLegal():
        pot_sudokus.append(new_p)
        i += 1
        if i >= n:
          return pot_sudokus
    # if we ran out of options, return whatever we found.
    return pot_sudokus

  def checkLegal(self):
    # Check that each row/col/box has at most 1 of each digit
    # Check sum of each rows and col is 45
    for i in range(1, 10):
      # for each digit, check row/col have at most 1 of that digit
      v = self.grid == i
      if v.sum(axis=0).max() > 1:
        return False
      if v.sum(axis=1).max() > 1:
        return False
      # Check 3x3 boxes
      for i in range(3):
        for j in range(3):
          box = v[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
          if sum(box.flatten()) > 1:
            return False
    return True


  def getGridLegalPossibilities(self):
    p = defaultdict(set)
    all9 = set(range(1, 10))  # set 1 to 9
    for r in range(9):
      for c in range(9):
        if self.grid[r, c] != 0:
          val = self.grid[r, c]
          # add this value to rows/cols/3x3
          for i in range(9):
            p[(i, c)].add(val)
            p[(r, i)].add(val)
          box_r = int(r / 3) * 3
          box_c = int(c / 3) * 3
          for i in range(3):
            for j in range(3):
              p[(box_r + i, box_c + j)].add(val)
          p[(r, c)] = all9
    # invert set
    possibilitySet = dict()
    min_length = 9
    for r in range(9):
      for c in range(9):
        if self.grid[r, c] == 0:
          possibilitySet[(r, c)] = all9 - p[(r, c)]
          l1 = 9 - len(p[(r, c)])
          if l1 < min_length:
            min_length = l1
    # order by least choices
    possibilitySet = OrderedDict(
      sorted(possibilitySet.items(), key=lambda item: len(item[1]))
    )
    return possibilitySet, min_length

  def ppgrid(self):
    """ Pretty print grid """
    msg = ""
    x = self.grid
    for i in range(9):
      msg += f"{x[i,0]} {x[i,1]} {x[i,2]} | {x[i,3]} {x[i,4]} {x[i,5]} | {x[i,6]} {x[i,7]} {x[i,8]}\n"
      if i in (2, 5):
        msg += "---------------------\n"
    return msg

  def __repr__(self):
    return f"Legal[{self.legal}]:Complete[{self.complete} / {self.getCompleteness()}]:Depth[{self.depth}]:\n{self.ppgrid()}"

  def __gt__(self, potsudoku2):
    p1 = self.getPriority()
    p2 = potsudoku2.getPriority()
    # lower priority is better
    if p1 == p2:
      # more complete is better
      return self.getCompleteness() < potsudoku2.getCompleteness()
    return p1 < p2
