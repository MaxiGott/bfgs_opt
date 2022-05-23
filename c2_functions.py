import typing
import copy
import numpy as np


class C2Function:
    def __init__(self, f, symbolic: bool = True):
        self.f = f
        self.h = 1e-5

    def gradient(self, x: typing.Union[float, typing.List[float]]):
        if isinstance(x, int):
            x = [x]

        gradient = np.array([None] * len(x))

        for i in range(len(x)):
            x_ = copy.deepcopy(x)
            x_[i] += self.h

            gradient[i] = (self.f(*x_) - self.f(*x)) / self.h
        return np.array(gradient)

    @staticmethod
    def sign(x):
        """
        Returns 1 if x is positive, else -1
        Function is needed because sign of numpy does return zero -> we only need 1, -1
        :param x: some value
        :return: 1, -1
        """
        if x == 0:
            return 1
        else:
            return np.sign(x)

    def hessian(self, x: typing.Union[float, typing.List[float], np.ndarray], round_dec: float = None):
        rows, cols = len(x), len(x)
        typ_x = 1
        h_ = self.h ** (1 / float(3))
        hessian = np.ones((rows, cols))
        for i in range(rows):
            x_i = copy.deepcopy(x)
            h_i = h_ * self.sign(x[i]) * np.max([x[i], typ_x])
            x_i[i] += h_i
            for j in range(cols):
                x_j = copy.deepcopy(x)
                h_j = h_ * self.sign(x[j]) * np.max([x[j], typ_x])
                x_j[j] += h_j

                x_ij = copy.deepcopy(x)
                x_ij[i] += h_i
                x_ij[j] += h_j

                hessian[i, j] = ((self(*x_ij) - self(*x_i)) - (self(*x_j) - self(*x))) / (h_i * h_j)

        if round_dec:
            hessian = np.round(hessian, decimals=round_dec)

        return hessian

    def __call__(self, *args, **kwargs):
        return self.f(*args)
