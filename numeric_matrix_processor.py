"""A numeric matrix processor that performs basic operations on matrix objects
defined by user input. The user chooses from a menu of possible operations,
is prompted to input matrix dimensions and values and the chosen operation is
performed.

Supported operations:

- Addition
- Subtraction
- Transposition
- Scalar multiplication
- Matrix multiplication
- Find the determinant
- Find the inverse
"""


class Matrix:
    """A class to represent a numeric matrix.

    ...

    Attributes
    ----------
    rows : int
        number of rows m in an m x n numeric matrix X
    columns : int
        number of columns n in an m x n numeric matrix X
    values : list
        nested list of real numbers representing rows
        in an m x n numeric matrix X

    Methods
    -------
    sum(other):
        returns a nested list representing the sum of 2 matrix objects if
        their dimensions are equal
    diff(other):
        returns a nested list representing the difference of 2 matrix objects
        if their dimensions are equal
    scalar_product(x):
        multiplies a matrix object by a constant and returns a nested
        list representing the product
    transpose():
        returns a nested list representing the transpose of a matrix object
    product(other):
        multiplies a matrix object (m1 x n1) by another matrix object (m2 x n2)
        where n1 == m2 and returns a nested list representing the product
    determinant():
        finds the determinant of an n x n square matrix and returns a float
    inverse():
        finds the inverse of a matrix
    """

    def __init__(self, rows, columns, values):
        """Constructs necessary attributes for the matrix object.

        Parameters
        ----------
            rows : int
                number of rows m in an m x n numeric matrix X
            columns : int
                number of columns n in an m x n numeric matrix X
            values : list
                nested list of real numbers representing rows
                in an m x n numeric matrix X
        """
        self.rows = rows
        self.columns = columns
        self.values = values

    def __repr__(self):
        """Returns a string representation of the class declaration used to
        implement an instance of the matrix object.
        """
        return 'Matrix(rows={}, columns={}, values={})'.format(self.rows,
                                                               self.columns,
                                                               self.values)

    def sum(self, other):
        """Prints the sum of 2 matrix objects if their dimensions are equal.

        Parameters
        ----------
        other : object of class Matrix

        Returns
        -------
        Nested list or string
        """
        if self.rows == other.rows and self.columns == other.columns:
            matrix_sum = [[self.values[i][j] + other.values[i][j] for j in
                           range(len(self.values[0]))] for i in range(len(
                               self.values))]
            return matrix_sum
        else:
            return 'The operation cannot be performed.\n'

    def diff(self, other):
        """Prints the difference of 2 matrix objects if their dimensions
        are equal.

        Parameters
        ----------
        other : object of class Matrix

        Returns
        -------
        Nested list or string
        """
        if self.rows == other.rows and self.columns == other.columns:
            matrix_diff = [[self.values[i][j] - other.values[i][j] for j in
                            range(len(self.values[0]))] for i in range(len(
                                self.values))]
            return matrix_diff
        else:
            return 'The operation cannot be performed.\n'

    def scalar_product(self, x):
        """Multiplies a matrix object by a constant and prints the product.

        Parameters
        ----------
        x : float
            real number used as a constant

        Returns
        -------
        Nested list
        """
        matrix_product = [[self.values[i][j] * x for j in
                           range(len(self.values[0]))] for i in range(len(
                               self.values))]
        return matrix_product

    def transpose(self):
        """Returns a nested list representing the transpose of a matrix
        object.

        Parameters
        ----------
        None

        Returns
        -------
        Nested list
        """
        matrix_transpose = [list(row) for row in zip(*self.values)]
        return matrix_transpose

    def transpose_side(self):
        """Returns a nested list representing a matrix transposed along its
        side diagonal. Not a mathematical operation. For instructional
        purposes only.

        Parameters
        ----------
        None

        Returns
        -------
        Nested list
        """
        transposed = self.transpose()
        transposed_side = [row[::-1] for row in transposed]
        return transposed_side[::-1]

    def transpose_vertical(self):
        """Returns a nested list representing a matrix reflected across a
        vertical line. Not a mathematical operation. For instructional
        purposes only.

        Parameters
        ----------
        None

        Returns
        -------
        Nested list
        """
        transpose_vertical = [row[::-1] for row in self.values]
        return transpose_vertical

    def transpose_horizontal(self):
        """Returns a nested list representing a matrix reflected across a
        horizontal line. Not a mathematical operation. For instructional
        purposes only.

        Parameters
        ----------
        None

        Returns
        -------
        Nested list
        """
        transpose_horizontal = self.values[::-1]
        return transpose_horizontal

    def product(self, other):
        """Mulitplies a matrix object (m1 x n1) by another matrix object
        (m2 x n2) where n1 == m2 and returns the product.

        Parameters
        ----------
        other : object of class Matrix

        Returns
        -------
        Nested list or string
        """

        def dot_product(row, col):
            """Returns the dot product of a given row and column.

            Parameters
            ----------
            row : list
                a row from matrix A
            col : list
                a column from matrix B

            Returns
            -------
            float
            """
            product = [row[i] * col[i] for i in range(len(row))]
            return sum(product)

        if self.columns == other.rows:
            other_transpose = other.transpose()
            product = [[dot_product(row, col) for col in other_transpose] for
                       row in self.values]
            return product
        else:
            return 'The operation cannot be performed.\n'

    def get_minor(self, matrix, i, j):
        """Returns the minor matrix of a given matrix.

        Parameters
        ----------
        matrix : list of lists
        i : int
        j : int

        Returns
        -------
        List of lists
        """
        return [row[:j] + row[j + 1:] for row in (
            matrix[:i] + matrix[i + 1:])]

    def get_determinant(self, matrix):
        """Finds the determinant of an n x n matrix via recursion.

        Parameters
        ----------
        matrix : list of lists

        Returns
        -------
        float
        """
        # Check dimensions
        if self.rows != self.columns:
            return 'Not a square matrix.'
        # Base case: determinant of a 2x2 matrix
        elif len(matrix) == 1:
            return matrix[0][0]
        # Recursive case: determinant of an n x n square matrix
        result = 0
        for j in range(len(matrix)):
            cofactor = (-1) ** j * matrix[0][j] * self.get_determinant(
                self.get_minor(matrix, 0, j))
            result += cofactor
        return result

    def determinant(self):
        """Returns the determinant of an n x n square matrix.

        Parameters
        ----------
        none

        Returns
        -------
        Float
        """
        return self.get_determinant(self.values)

    def inverse(self):
        """Return the inverse of a matrix.

        Parameters
        ----------
        None

        Returns
        -------
        List of lists, else None if no inverse
        """
        if self.determinant() == 0:
            return None
        else:
            cofactors = []
            for i in range(len(self.values)):
                cofactor_row = []
                for j in range(len(self.values[0])):
                    minor = [row[:j] + row[j + 1:] for row in (
                        self.values[:i] + self.values[i + 1:])]
                    cofactor_row.append(((-1) ** (i + j))
                                        * self.get_determinant(minor))
                cofactors.append(cofactor_row)
            cofactors_transposed = [list(row) for row in zip(*cofactors)]
            return [[cofactors_transposed[i][j] * (1 / self.determinant())
                     for j in range(len(cofactors_transposed[0]))]
                    for i in range(len(cofactors_transposed))]


def main():
    state = 'menu'
    while state != 'exiting':   # Initialize menu
        if state == 'menu':     # Main menu
            print('1. Add matrices'
                  + '\n2. Multiply matrix by a constant'
                  + '\n3. Multiply matrices'
                  + '\n4. Transpose matrix'
                  + '\n5. Calculate a determinant'
                  + '\n6. Inverse matrix'
                  + '\n0. Exit')
            action = input('Your choice: ').strip()
            if action == '1':
                state = 'addition'
            elif action == '2':
                state = 'scalar product'
            elif action == '3':
                state = 'matrix product'
            elif action == '4':
                state = 'transpose'
            elif action == '5':
                state = 'determinant'
            elif action == '6':
                state = 'inverse'
            elif action == '0':
                state = 'exiting'
        elif state == 'addition':   # Matrix addition menu
            matrix_1_dims = list(map(int, input(
                'Enter size of first matrix: ').split()))
            print('Enter first matrix:')
            matrix_1_input = [list(map(float, input().split())) for i
                              in range(matrix_1_dims[0])]
            matrix_2_dims = list(map(int, input(
                'Enter size of second matrix: ').split()))
            print('Enter second matrix:')
            matrix_2_input = [list(map(float, input().split())) for i
                              in range(matrix_2_dims[0])]
            matrix_1 = Matrix(matrix_1_dims[0], matrix_1_dims[1],
                              matrix_1_input)
            matrix_2 = Matrix(matrix_2_dims[0], matrix_2_dims[1],
                              matrix_2_input)
            print('The result is:')
            for row in matrix_1.sum(matrix_2):
                print(*row)
            print()
            state = 'menu'
        elif state == 'scalar product':    # Scalar multiplication menu
            matrix_dims = list(map(int, input('Enter size of matrix: ')
                                   .split()))
            print('Enter matrix:')
            matrix_input = [list(map(float, input().split()))
                            for i in range(matrix_dims[0])]
            CONSTANT = int(input('Enter constant: '))
            matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
            print('The result is')
            for row in matrix.scalar_product(CONSTANT):
                print(*row)
            print()
            state = 'menu'
        elif state == 'matrix product':     # Matrix multiplication menu
            matrix_1_dims = list(map(int, input(
                'Enter size of first matrix: ').split()))
            print('Enter first matrix:')
            matrix_1_input = [list(map(float, input().split())) for i
                              in range(matrix_1_dims[0])]
            matrix_2_dims = list(map(int, input(
                'Enter size of second matrix: ').split()))
            print('Enter second matrix:')
            matrix_2_input = [list(map(float, input().split())) for i
                              in range(matrix_2_dims[0])]
            matrix_1 = Matrix(matrix_1_dims[0], matrix_1_dims[1],
                              matrix_1_input)
            matrix_2 = Matrix(matrix_2_dims[0], matrix_2_dims[1],
                              matrix_2_input)
            print('The result is:')
            for row in matrix_1.product(matrix_2):
                print(*row)
            print()
            state = 'menu'
        elif state == 'transpose':      # Matrix transposition menu
            print('\n1. Main diagonal'
                  + '\n2. Side diagonal'
                  + '\n3. Vertical line'
                  + '\n4. Horizonal line')
            action = input('Your choice ').strip()
            if action == '1':
                matrix_dims = list(map(int, input(
                    'Enter matrix size: ').split()))
                print('Enter matrix:')
                matrix_input = [list(map(float, input().split())) for i
                                in range(matrix_dims[0])]
                matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
                print('The result is:')
                for row in matrix.transpose():
                    print(*row)
            elif action == '2':
                matrix_dims = list(map(int, input(
                    'Enter matrix size: ').split()))
                print('Enter matrix:')
                matrix_input = [list(map(float, input().split())) for i
                                in range(matrix_dims[0])]
                matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
                print('The result is:')
                for row in matrix.transpose_side():
                    print(*row)
            elif action == '3':
                matrix_dims = list(map(int, input(
                    'Enter matrix size: ').split()))
                print('Enter matrix:')
                matrix_input = [list(map(float, input().split())) for i
                                in range(matrix_dims[0])]
                matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
                print('The result is:')
                for row in matrix.transpose_vertical():
                    print(*row)
            elif action == '4':
                matrix_dims = list(map(int, input(
                    'Enter matrix size: ').split()))
                print('Enter matrix:')
                matrix_input = [list(map(float, input().split())) for i
                                in range(matrix_dims[0])]
                matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
                print('The result is:')
                for row in matrix.transpose_horizontal():
                    print(*row)
            print()
            state = 'menu'
        elif state == 'determinant':      # Determinant menu
            matrix_dims = list(map(int, input('Enter size of matrix: ')
                                   .split()))
            print('Enter matrix:')
            matrix_input = [list(map(float, input().split()))
                            for i in range(matrix_dims[0])]
            matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
            print('The result is:')
            print(matrix.determinant())
            print()
            state = 'menu'
        elif state == 'inverse':        # Inverse menu
            matrix_dims = list(map(int, input('Enter matrix size: ')
                                   .split()))
            print('Enter matrix:')
            matrix_input = [list(map(float, input().split()))
                            for i in range(matrix_dims[0])]
            matrix = Matrix(matrix_dims[0], matrix_dims[1], matrix_input)
            matrix_inverse = matrix.inverse()
            if matrix.inverse() is None:
                print("This matrix doesn't have an inverse.")
                state = 'menu'
            else:
                print('The result is:')
                for row in matrix_inverse:
                    print(*row)
                print()
                state = 'menu'


if __name__ == '__main__':
    main()
