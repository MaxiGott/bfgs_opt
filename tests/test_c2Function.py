from C2Functions import C2Function
import unittest


def x_squared(x):
    return x**2


def x_y_squared(x, y):
    return x**2 + y**2


def c2_x_squared():
    return C2Function(x_squared)


def c2_x_y_squared():
    return C2Function(x_y_squared)


class C2Tests(unittest.TestCase):

    def test_gradient_1_dim(self):
        func = c2_x_squared()
        grad = func.gradient(0)
        self.assertEqual(-5e-10, grad[0])

    def test_gradient_2_dim(self):
        func = c2_x_y_squared()
        grad = func.gradient([0, 0])
        self.assertEqual(-5e-10, grad[0])
        self.assertEqual(5e-10, grad[1])
