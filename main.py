import time
from bfgs import BFGS
import scipy.optimize

def func(x):
    return x**2


def func_2(x, y):
    return x**2 + y**2


def rosenbrock_2d(x, y, a=1, b=1):
    return (a-x)**2 + b*(y-x**2)**2


def rosenbrock_nd(*args, a=1, b=1):
    sum = 0
    for i in range(len(args) - 1):
        sum += b*(args[i+1]-args[i]**2)**2+(a-args[i])**2
    return sum


if __name__ == '__main__':
    x0 = [5, 4]
    some_func = rosenbrock_nd

    # scipy
    start = time.time()
    erg = scipy.optimize.minimize(rosenbrock_nd, x0, method="BFGS")
    print(f"Duration scipy: {time.time()-start}")

    # own implementation
    start = time.time()
    x = BFGS(f=some_func, x0=x0, eps=1e-4, plotting=False).minimize(trust_region=True)
    print("Duration: " + str(time.time() - start))

    print(f"Minimum: {x}, Value at function: {some_func(*x)}")#
