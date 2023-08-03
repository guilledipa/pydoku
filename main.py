import cv2
import pyautogui

from sudoku_solver import SudokuComHandler
from sudoku_solver import SudokuSolver


def main():
    sudoku_com = SudokuComHandler("solve web sudoku puzzles")
    solver = SudokuSolver(sudoku_com.sudoku)
    solved = solver.solve()
    sudoku_com.solve_on_website(solved)


if __name__ == '__main__':
    main()