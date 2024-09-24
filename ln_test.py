from linear_regression import LinearRegression
import numpy as np


def ln_test_1():
    x = np.linspace(0, 150, 200)
    y = 0.72 * x + 7
    noise = np.random.normal(0, 6, x.shape)
    y = y + noise
    ln = LinearRegression(x, y)
    ln.plot_line()
    ln.print_metrics()
    ln.plot_diagnostics()


ln_test_1()
