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

    def minimize(self, line_search=False, trust_region=False):
        # calculate hessian matrix as start
        h_k = np.linalg.inv(self.f.hessian(x=self.x0))
        # initialize starting point
        x_k = self.x0
        # counter
        k = 0
        # while gradient big enough and not more than max_iter iterations

        search_method = None
        if (line_search and trust_region) or not (line_search and trust_region):
            print("Please select exactly one method, either line_search or trust_region")

        while np.linalg.norm(self.f.gradient(x_k), 2) > self.eps or k > self.max_iter:
            # counter + 1
            k += 1
            grad = self.f.gradient(x_k)

            # compute next iteration point
            if line_search:
                x_k_ = self.line_search(self.alpha, x_k, h_k)
            else:
                x_k_ = self.trust_region(x_k, h_k)

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
        p = 1 / np.dot(y, s)
        # create I
        I = np.diag([1] * len(hessian))
        # partial calculation
        x = I - p * np.outer(s, y)
        y = I - p * np.outer(y, s)
        z = p * np.outer(s, s)

        hessian_ = np.dot(np.dot(x, hessian), y) + z
        return hessian_

    def trust_region(self, x, h, delta=1e-1, delta_max: float = 1, v_=1e-1):
        delta_k = delta
        while True:
            s_k = self._search_direction(x, h, delta=delta_k)
            v_k = (self.f(*x) - self.f(*(x + s_k))) / (self._mc(x, s=np.zeros(len(x)), h=h) - self._mc(x, s=s_k, h=h))

            if v_k < 1 / 4:
                delta_k = 1 / 2 * np.linalg.norm(s_k)
            elif v_k > 3 / 4 and np.linalg.norm(s_k) == delta_k:
                delta_k = min(2 * delta_k, delta_max)
            else:
                break

        return x + s_k

    def line_search(self, l, x, h):
        grad_f = self.f.gradient(x)
        # compute search direction
        p = -(np.dot(h, grad_f))

        m = 1
        while not self.wolfe_condition(l, x, p) and l > 1e-6:
            l *= 0.5
            m += 1
        return x + l * p

    def wolfe_condition(self, l: float, x: np.ndarray, p: np.ndarray, alpha=0.1, beta=0.9) -> bool:
        first_cond = False
        second_cond = False

        x_ = x + l * p

        if self.f(*x_) <= self.f(*x) + alpha * l * np.dot(self.f.gradient(x), p):
            first_cond = True

        if np.dot(self.f.gradient(x_), p) >= beta * np.dot(self.f.gradient(x), p):
            second_cond = True

        return first_cond and second_cond

    def _mc(self, x, s, h):
        return self.f(*x) + np.dot(self.f.gradient(x), s) + 1 / 2 * np.dot(s, np.dot(h, s))

    def _search_direction(self, x, h, delta=0.01):
        grad_f = self.f.gradient(x)
        t_ = np.linalg.norm(grad_f, 2) ** 3 / (delta * np.dot(grad_f, np.dot(h, grad_f)))
        t = min(t_, 1)
        s_c = -delta * grad_f / np.linalg.norm(grad_f, 2)
        return t * s_c
