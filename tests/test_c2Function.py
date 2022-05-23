from c2_functions import C2Function
import unittest
import numpy as np

def x_squared(x):
    return x ** 2


def x_y_squared(x, y):
    return x ** 2 + y ** 2


def rosenbrock_nd(*args, a=1, b=1):
    sum = 0
    for i in range(len(args) - 1):
        sum += b * (args[i + 1] - args[i] ** 2) ** 2 + (a - args[i]) ** 2
    return sum


def c2_x_squared():
    return C2Function(x_squared)


def c2_x_y_squared():
    return C2Function(x_y_squared)


def c2_rosenbrock_nd():
    return C2Function(rosenbrock_nd)


class C2Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_squared = c2_x_squared()
        self.xy_squared = c2_x_y_squared()
        self.rosenbrock = c2_rosenbrock_nd()

    def test_gradient_1_dim(self):
        grad = self.x_squared.gradient(0)
        self.assertEqual(0, round(grad[0]))

    def test_gradient_2_dim(self):
        grad = self.xy_squared.gradient([0, 0])

        self.assertEqual(0, round(grad[0]))
        self.assertEqual(0, round(grad[1]))

    def test_hessian_2_dim(self):
        h = self.xy_squared.hessian(x=[0, 0])
        self.assertEqual(2, round(h[0][0]))

    def test_hessian_2_dim_rosenbrock(self):
        # Minimum at 1, 1
        h = self.rosenbrock.hessian(x=[1, 1])
        self.assertEqual(2, round(h[1][1]))

