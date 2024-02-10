#!/usr/bin/env python3
from .utils import random_float, sigmoid

class Matrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for _ in range(0, self.cols)] for _ in range(0, self.rows)]

    @staticmethod
    def from_list(input_list):
        """Create a new Matrix from a input list, returns new Matrix"""
        rows = len(input_list)
        cols = len(input_list[0])
        return_matrix = Matrix(rows, cols)
        for i in range(0, rows):
            for j in range(0, cols):
                return_matrix.data[i][j] = input_list[i][j]
        return return_matrix


    def randomize(self, start=0, stop=1):
        """Randomize all elements inside of the matrix, can be configured with start and stop"""
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.data[row][col] = random_float(start, stop)

    def fill(self, num):
        """Fills all elements of the matrix with the given number"""
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.data[row][col] = num

    def dot(self, matrix):
        """Matrix multiplication of the Matrix and the provided Matrix. Will return the result as a new Matrix"""
        assert self.cols == matrix.rows, "Multiplication Error: Number of cols have to match with number of rows"

        return_matrix = Matrix(self.rows, matrix.cols)

        for row in range(0, return_matrix.rows):
            for col in range(0, return_matrix.cols):
                for i in range(0, self.cols):
                    return_matrix.data[row][col] += self.data[row][i] * matrix.item_at(i, col)

        return return_matrix


    def sum(self, matrix):
        """Sums matrix with the given matrix. Result will be stored in the object"""
        assert self.rows == matrix.rows, "Addition Error: number of rows have to match"
        assert self.cols == matrix.cols, "Addition Error: number of cols have to match"

        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.data[row][col] += matrix.item_at(row, col)

    def sigmoid(self):
        """All values of the matrix will be calculated with the sigmoid function"""
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.data[row][col] = sigmoid(self.data[row][col])

    def sub_matrix(self, rows, cols, start_row=0, start_col=0):
        """Returns a submatrix as new object"""
        return_matrix = Matrix(rows, cols)
        for i, row in enumerate(range(start_row, rows + start_row)):
            for j, col in enumerate(range(start_col, start_col + cols)):
                return_matrix.data[i][j] = self.data[row][col]
        return return_matrix

    def item_at(self, row, col):
        """Returns item at requested position"""
        return self.data[row][col]

    def copy(self, matrix):
        """Copy given matrix to the object"""
        #print(matrix)
        #print(self)
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.data[row][col] = matrix.item_at(row, col)

    def __mul__(self, other):
        """* Overriden for matrix. Therefore multiplication can be written as m1*m2"""
        return self.dot(other)

    def __add__(self, other):
        """+ Operator overriden for matrix. Therefore addition can be written as m1+m2"""
        self.sum(other)

    def __str__(self):
        """Returns string representation of the Matrix"""
        return_string = "[\n"
        for row in self.data:
            return_string += "\t"
            for col in row:
                return_string += str(col) + " "
            return_string += "\n"
        return_string += "]"
        return return_string
