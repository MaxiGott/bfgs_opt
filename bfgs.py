import typing
import numpy as np
from c2_functions import C2Function


class BFGS:
    f: C2Function

    def __init__(self, f, x0: typing.List[int], alpha: float = 1, max_iter: int = 1e5, eps: float = 1e-5):
        self.eps = eps
        self.max_iter = max_iter
        # select a start val
        self.x0 = x0
        self.alpha = alpha

        if not isinstance(f, C2Function):
            self.f = C2Function(f)
        else:
            self.f = f

    def minimize(self):
        # calculate hessian matrix as start
        h_k = np.linalg.inv(self.f.hessian(x=self.x0))
        # initialize starting point
        x_k = self.x0
        # counter
        k = 0
        # while gradient big enough and not more than max_iter iterations
        while np.linalg.norm(self.f.gradient(x_k), 2) > self.eps or k > self.max_iter:
            # counter + 1
            k += 1
            grad = self.f.gradient(x_k)

            # compute search direction
            p_k = -(np.dot(h_k, grad))

            # compute step wide
            a_k = self._line_search(self.alpha, x_k, p_k)

            # compute next iteration point
            x_k_ = x_k + a_k*p_k

            # difference to first point
            s_k = x_k_ - x_k
            # difference between gradients (like momentum)
            next_grad = self.f.gradient(x_k_)
            y_k = next_grad - grad

            # update hessian matrix
            h_k_ = self._update_hessian(h_k, s=s_k, y=y_k)
            # update x_k
            x_k = x_k_

            # update hessian
            h_k = h_k_

        print(f"Finished after: {k} steps. Minimum value: {self.f(*x_k)} at point {x_k}")
        return x_k

    @staticmethod
    def _update_hessian(hessian, s, y):
        """
        :param hessian: Hessian matrix
        :param s: difference between the iteration points
        :param y: difference between the gradients
        :return: updated hessian matrice
        """
        p = 1/np.dot(y, s)
        # create I
        I = np.diag([1] * len(hessian))
        # partial calculation
        x = I - p*np.outer(s, y)
        y = I - p*np.outer(y, s)
        z = p*np.outer(s, s)

        hessian_ = np.dot(np.dot(x, hessian), y) + z
        return hessian_

    def wolfe_condition(self, l: float, x: np.ndarray, p: np.ndarray, alpha=0.1, beta=0.9) -> bool:
        first_cond = False
        second_cond = False

        x_ = x + l * p

        if self.f(*x_) <= self.f(*x) + alpha * l * np.dot(self.f.gradient(x), p):
            first_cond = True

        if np.dot(self.f.gradient(x_), p) >= beta * np.dot(self.f.gradient(x), p):
            second_cond = True

        return first_cond and second_cond

    def _line_search(self, l, x, p):
        m = 1
        while not self.wolfe_condition(l, x, p) and l > 1e-6:
            l *= 0.5
            m += 1
        return l