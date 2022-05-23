import typing
import copy
import numpy as np


class C2Function:
    def __init__(self, f, symbolic: bool = True):
        self.f = f
        self.h = 1e-5

    def gradient(self, x: typing.Union[int, typing.List[float]]):
        if isinstance(x, int):
            x = [x]

        gradient = np.array([None] * len(x))

        for i in range(len(x)):
            x_ = copy.deepcopy(x)
            x_[i] += self.h

            gradient[i] = (self.f(*x_) - self.f(*x)) / self.h
        return np.array(gradient)

    def hessian(self, x: typing.Union[int, typing.List[int]], round_dec: int = None):
        rows, cols = len(x), len(x)
        typ_x = 1
        h_ = self.h ** (1 / float(3))
        hessian = np.ones((rows, cols))
        for i in range(rows):
            x_i = copy.deepcopy(x)
            h_i = h_ * np.sign(x[i]) * np.max([x[i], typ_x])
            x_i[i] += h_i
            for j in range(cols):
                x_j = copy.deepcopy(x)
                h_j = h_ * np.sign(x[j]) * np.max([x[j], typ_x])
                x_j[j] += h_j

                x_ij = copy.deepcopy(x)
                x_ij[i] += h_i
                x_ij[j] += h_j

                hessian[i, j] = ((self.f(*x_ij) - self.f(*x_i)) - (self.f(*x_j) - self.f(*x))) / (h_i * h_j)

        if round_dec:
            hessian = np.round(hessian, decimals=round_dec)

        return hessian

    def __call__(self, *args, **kwargs):
        return self.f(*args)
