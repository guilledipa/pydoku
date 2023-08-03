import cv2
import numpy
import pandas
import pyautogui
import skimage
from sklearn.neighbors import KNeighborsClassifier
import time

class SudokuComHandler:

    def __init__(self, window_title):
        self.window_title = window_title
        self.square_contour = self.get_square_contour()
        self.sudoku = self.extract_sudoku()

    def get_square_contour(self):
        pyautogui.getWindowsWithTitle(self.window_title)[0].activate()
        time.sleep(1)
        full_res_screenshot = pyautogui.screenshot()
        self.screenshot = cv2.cvtColor(numpy.array(full_res_screenshot), cv2.COLOR_RGB2BGR)
        preprocessed = self.coarse_preprocess(self.screenshot)
        return self.find_sudoku_contour(preprocessed)
    
    def extract_sudoku(self):
        if self.square_contour is None:
            print("No sudoku found")
            return 1
        cropped_grid = self.crop_grid(self.screenshot, self.square_contour)
        squares_images = self.split_grid(cropped_grid)
        return self.squares_images_to_sudoku(squares_images)

    def coarse_preprocess(self, screenshot):
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # For compression purposes
        thresh  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def find_sudoku_contour(self, preprocessed_image):
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        squares = []
        for contour in contours:
            if self.is_square(contour):
                squares.append(contour)
        squares = sorted(squares, key=cv2.contourArea, reverse=True)
        if len(squares) == 0:
            return None
        return squares[0]

    def is_square(self, contour):
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return len(approx) == 4 and abs(aspect_ratio - 1) < 0.1

    def crop_grid(self, screenshot, square):
        x, y, w, h = cv2.boundingRect(square)
        cropped = screenshot[y:y+h, x:x+w]
        return cropped

    def split_grid(self, cropped_grid):
        img = self.coarse_preprocess(cropped_grid)
        img = skimage.segmentation.clear_border(img)
        img = 255 - img
        height, _ = img.shape
        square_size = height // 9
        squares = []
        for i in range(9):
            for j in range(9):
                square_img = img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size]
                square_img = cv2.resize(square_img, (55, 55), interpolation=cv2.INTER_AREA)
                squares.append(square_img)
        return squares

    # -- Machine learning model
    def squares_images_to_sudoku(self, squares_images):
        knn = self.create_knn_model()
        sudoku = numpy.zeros((81), dtype=int)
        for i, image in enumerate(squares_images):
            sudoku[i] = self.predict_digit(image, knn)
        return sudoku.reshape(9, 9)

    def predict_digit(self, img, knn):
        img_vec = img.reshape(1, -1)
        prediction = knn.predict(img_vec)[0]
        return prediction

    def create_knn_model(self):
        df = pandas.read_csv("dataset.csv")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x, y)
        return knn

    def solve_on_website(self, solved):
        x, y, w, h = cv2.boundingRect(self.square_contour)
        square_size = h // 9
        for i in range(9):
            for j in range(9):
                pyautogui.click(x + j*square_size + square_size//2, y + i*square_size + square_size//2, _pause=False)
                pyautogui.press(str(solved[i, j]), _pause=False)

class SudokuSolver:

    def __init__(self, sudoku):
        self.bitmap = numpy.ones((9, 9, 9), dtype=bool)
        self.sudoku = numpy.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                if sudoku[i][j] != 0:
                    self.place_number(i, j, sudoku[i][j])

    def solve(self):
        start = time.time()
        self.backtrack()
        end = time.time()
        self.solve_time = end - start
        return self.sudoku
    
    def backtrack(self):
        if self.is_solved():
            return self.sudoku
        i, j = self.least_options_cell()
        for number in range(1, 10):
            if self.can_place_number(i, j, number):
                curr_bitmap = self.bitmap.copy()
                curr_sudoku = self.sudoku.copy()
                self.place_number(i, j, number)
                self.trivial_moves()
                if self.backtrack() is not None:
                    return self.sudoku
                self.bitmap = curr_bitmap
                self.sudoku = curr_sudoku
        return None
    
    def is_solved(self):
        return numpy.sum(self.sudoku == 0) == 0
    
    def least_options_cell(self):
        cell_sums = numpy.sum(self.bitmap, axis=2)
        cell_sums[cell_sums == 0] = 10
        min_sum_indices = numpy.unravel_index(numpy.argmin(cell_sums), cell_sums.shape)
        return min_sum_indices
    
    def can_place_number(self, i, j, number):
        return self.bitmap[i][j][number-1]

    def place_number(self, i, j, number):
        self.sudoku[i][j] = number
        self.bitmap[i][j] = numpy.zeros(9, dtype=bool)
        for k in range(9):
            self.bitmap[i][k][number-1] = False
            self.bitmap[k][j][number-1] = False
        for k in range(3):
            for l in range(3):
                self.bitmap[i//3*3+k][j//3*3+l][number-1] = False
    
    def trivial_moves(self):
        changed = True
        while changed:
            changed = False
            for i in range(9):
                for j in range(9):
                    if self.is_trivial_cell(i, j):
                        changed = True
                        self.place_number(i, j, numpy.argmax(self.bitmap[i][j]) + 1)

    def is_trivial_cell(self, i, j):
        return self.sudoku[i][j] == 0 and numpy.sum(self.bitmap[i][j]) == 1
