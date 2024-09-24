import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class LinearRegression:
    def __init__(self, x, y):
        self.independent_data = x
        self.dependent_data = y
        self.n_data_points = np.size(x)
        self.mean_x = np.mean(x)
        self.mean_y = np.mean(y)
        self.__calculate_param()


    def __calculate_param(self):

        x_minus_mean_x = self.independent_data - self.mean_x
        y_minus_mean_y = self.dependent_data - self.mean_y

        xy_product = x_minus_mean_x * y_minus_mean_y
        self.SSxy = np.sum(xy_product)


        xx_product = x_minus_mean_x ** 2
        self.SSxx = np.sum(xx_product)

        self.slope = self.SSxy/self.SSxx
        self.intercept = self.mean_y - self.slope * self.mean_x

        print("B0: %.2f and B1: %.2f" % (self.intercept, self.slope))

        self.predicted_y = self.intercept + self.slope * self.independent_data
        self.residuals = self.dependent_data - self.predicted_y

    def result(self):
        return self.intercept, self.slope

    def plot_line(self):
        plt.scatter(self.independent_data, self.dependent_data, color="m",
                    marker="o", s=30)

        plt.plot(self.independent_data, self.predicted_y, color="g")

        # putting labels
        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()

    def print_metrics(self):
        absolute_error = np.sum(np.abs(self.dependent_data - self.predicted_y))
        mae = absolute_error/self.n_data_points

        squared_error = np.sum((self.dependent_data - self.predicted_y)**2)
        mse = squared_error/self.n_data_points

        rmse = np.sqrt(mse)

        tss = np.sum((self.dependent_data - self.mean_y )**2)
        r_square = 1 - (squared_error/tss)

        x_shape = x.ndim

        adjusted_r_square = 1 - ((1 - r_square)/(self.n_data_points - x_shape - 1)) * (self.n_data_points - 1)



        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2: {r_square}")
        print(f"Adjusted R2: {adjusted_r_square}")

    def plot_diagnostics(self):
        # 1. Residuals vs Fitted (for homoscedasticity)
        plt.scatter(self.predicted_y, self.residuals, color="blue", edgecolor='k')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title("Residuals vs Fitted Values")
        plt.show()

        # 2. Normal Q-Q plot
        stats.probplot(self.residuals, dist="norm", plot=plt)
        plt.title("Normal Q-Q Plot")
        plt.show()

        # 3. Histogram of Residuals
        plt.hist(self.residuals, bins=20, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.show()

        # 4. Residuals vs Leverage Plot (to detect influential points)
        leverage = (self.independent_data - self.mean_x) ** 2 / np.sum((self.independent_data - self.mean_x) ** 2)
        plt.scatter(leverage, self.residuals, color="blue", edgecolor='k')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Leverage')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Leverage')
        plt.show()




x = np.linspace(0, 150, 200)
y = 0.72 * x + 7
noise = np.random.normal(0, 6, x.shape)
y = y + noise
ln = LinearRegression(x, y)
ln.plot_line()
ln.print_metrics()
ln.plot_diagnostics()

















