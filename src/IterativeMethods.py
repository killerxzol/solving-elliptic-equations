import numpy as np
import pandas as pd


def get_grid(function, x_step, y_step, x_nodes, y_nodes):
    return np.array([[function(i * x_step, j * y_step) for j in range(y_nodes + 1)] for i in range(x_nodes + 1)])


def get_min_eigenvalue(x_step, y_step, x_len, y_len, c1, d1):
    return (c1 * 4 / (x_step * x_step) * np.sin(np.pi * x_step / (2 * x_len)) ** 2 +
            d1 * 4 / (y_step * y_step) * np.sin(np.pi * y_step / (2 * y_len)) ** 2)


def get_max_eigenvalue(x_step, y_step, x_len, y_len, c2, d2):
    return (c2 * 4 / (x_step * x_step) * np.cos(np.pi * x_step / (2 * x_len)) ** 2 +
            d2 * 4 / (y_step * y_step) * np.cos(np.pi * y_step / (2 * y_len)) ** 2)


def get_spectral_radius(min_eigenvalue, max_eigenvalue):
    return (max_eigenvalue - min_eigenvalue) / (max_eigenvalue + min_eigenvalue)


def diff_norm(m1, m2):
    return np.max(np.abs(m1 - m2))


class IterativeMethods:

    def __init__(self, u_function, f_function, p_function, q_function, constants, x_len, y_len, x_nodes, y_nodes, eps):
        """
        :param u_function: exact solution (only boundary values are needed for the approximate solution)
        :param f_function: F function
        :param p_function: p function
        :param q_function: q function
        :param constants: [c1, c2, d1, d2], where [0 < c1 <= p(x, y) <= c2], [0 < d1 <= q(x, y) <= d2]
        :param x_len: domain x: [0, x_len]
        :param y_len: domain y: [0, y_len]
        :param x_nodes: number of nodes in the grid by x
        :param y_nodes: number of nodes in the grid by y
        :param eps: iteration method stop condition
        """
        self.__p_function = p_function
        self.__q_function = q_function
        self.__x_nodes = x_nodes
        self.__y_nodes = y_nodes
        self.__x_step = x_len / x_nodes
        self.__y_step = y_len / y_nodes
        self._x_step_square = self.__x_step * self.__x_step
        self._y_step_square = self.__y_step * self.__y_step
        self.__eps = eps
        self.__u_grid = get_grid(u_function, self.__x_step, self.__y_step, x_nodes, y_nodes)
        self._f_grid = get_grid(f_function, self.__x_step, self.__y_step, x_nodes, y_nodes)
        self.__min_eigenvalue = get_min_eigenvalue(self.__x_step, self.__y_step, x_len, y_len, constants[0], constants[2])
        self.__max_eigenvalue = get_max_eigenvalue(self.__x_step, self.__y_step, x_len, y_len, constants[1], constants[3])
        self.__spectral_radius = get_spectral_radius(self.__min_eigenvalue, self.__max_eigenvalue)
        self.__total_iterations = 0
        self.__first_approximation = self.__dirichlet_boundary_condition(np.zeros((self.__x_nodes + 1, self.__y_nodes + 1)))
        self._last_approximation = np.copy(self.__first_approximation)
        self._current_approximation = np.copy(self.__first_approximation)
        self.__last_last_approximation = np.copy(self.__first_approximation)
        self.__estimate_table = pd.DataFrame(columns=['||F-AU^k||', 'rel.d.', '||U^k - U_*||',
                                                      'rel.error', 'a_post.est.', 'sp.rad._k'])

    def _solve(self, method_approximation, method_parameter=None):
        """
        Finds an approximation of the matrix U.
        :param method_approximation: approximation function of the chosen iterative method
        :param method_parameter: parameter used in method calculations
        :return: None
        """
        while self.__relative_error() > self.__eps:
            self.__total_iterations = self.__total_iterations + 1
            for i in range(1, self.__x_nodes):
                for j in range(1, self.__y_nodes):
                    if method_parameter:
                        self._current_approximation[i, j] = method_approximation(i, j, method_parameter)
                    else:
                        self._current_approximation[i, j] = method_approximation(i, j)
            self.__write_estimates()
            if self.__total_iterations > 1:
                self.__last_last_approximation = np.copy(self._last_approximation)
            self._last_approximation = np.copy(self._current_approximation)

    def get_solution(self):
        """
        Returns an approximate gridded solution of an elliptic equation.
        :return: numpy.array
        """
        return self._current_approximation

    def __dirichlet_boundary_condition(self, grid):
        """
        Writes the values of the Dirichlet problem to the boundary nodes.
        :return: numpy.array
        """
        for i in range(0, self.__x_nodes + 1):
            grid[i, 0] = self.__u_grid[i, 0]
            grid[i, self.__y_nodes] = self.__u_grid[i, -1]
        for j in range(1, self.__y_nodes):
            grid[0, j] = self.__u_grid[0, j]
            grid[self.__x_nodes, j] = self.__u_grid[-1, j]
        return grid

    def _get_p_values(self, i, j):
        """
        Returns the values of p function in intermediate nodes.
        :return: list
        """
        return [self.__p_function(self.__x_step * (i - 1 / 2), self.__y_step * j),
                self.__p_function(self.__x_step * (i + 1 / 2), self.__y_step * j)]

    def _get_q_values(self, i, j):
        """
        Returns the values of q function in intermediate nodes.
        :return: list
        """
        return [self.__q_function(self.__x_step * i, self.__y_step * (j - 1 / 2)),
                self.__q_function(self.__x_step * i, self.__y_step * (j + 1 / 2))]

    def __get_lu_grid(self, approximation):
        """
        Returns the value of the operator applied to the approximation grid (i.e. Lu).
        :param approximation: approximate gridded solution
        :return: numpy.array
        """
        lu_grid = np.copy(self.__first_approximation)
        for i in range(1, self.__x_nodes):
            for j in range(1, self.__y_nodes):
                p_values = self._get_p_values(i, j)
                q_values = self._get_q_values(i, j)
                lu_grid[i, j] = (p_values[1] * (approximation[i + 1, j] - approximation[i, j]) / self._x_step_square -
                                 p_values[0] * (approximation[i, j] - approximation[i - 1, j]) / self._x_step_square +
                                 q_values[1] * (approximation[i, j + 1] - approximation[i, j]) / self._y_step_square -
                                 q_values[0] * (approximation[i, j] - approximation[i, j - 1]) / self._y_step_square)
        return lu_grid

    # - - -

    # - - - estimate functions - - -

    def __residual_norm(self, approximation):
        """
        Returns the approximation residual norm.
        :return: numpy.float64
        """
        return diff_norm(self._f_grid[1:-1, 1:-1], -self.__get_lu_grid(approximation)[1:-1, 1:-1])

    def __relative_residual_norm(self):
        """
        Returns the relative residual of the current approximation.
        :return: numpy.float64
        """
        return (diff_norm(self._f_grid[1:-1, 1:-1], -self.__get_lu_grid(self._current_approximation)[1:-1, 1:-1]) /
                diff_norm(self._f_grid[1:-1, 1:-1], -self.__get_lu_grid(self.__first_approximation)[1:-1, 1:-1]))

    def __absolute_error_rate(self):
        """
        Returns the absolute error rate of the current approximation.
        :return: numpy.float64
        """
        return diff_norm(self.__u_grid, self._current_approximation)

    def __relative_error(self):
        """
        Returns the relative error of the current approximation.
        :return: numpy.float64
        """
        return diff_norm(self.__u_grid, self._current_approximation) / diff_norm(self.__u_grid, self.__first_approximation)

    def __a_posteriori_estimate(self):
        """
        Returns the error estimate of the current approximation.
        :return: numpy.float64
        """
        return diff_norm(self._last_approximation, self._current_approximation) * self.__spectral_radius / (1 - self.__spectral_radius)

    def __approach_to_spectral_radius(self):
        """
        Returns the current approximation to the spectral radius of the matrix.
        :return: numpy.float64
        """
        return (diff_norm(self._last_approximation, self._current_approximation) /
                diff_norm(self.__last_last_approximation, self._last_approximation))

    def __write_estimates(self):
        """
        Writes the obtained values of estimates to the table.
        :return: None
        """
        est1 = self.__residual_norm(self._current_approximation)
        est2 = self.__relative_residual_norm()
        est3 = self.__absolute_error_rate()
        est4 = self.__relative_error()
        est5 = self.__a_posteriori_estimate()

        if self.__total_iterations > 2:
            est6 = self.__approach_to_spectral_radius()
        else:
            est6 = None

        self.__estimate_table.loc[len(self.__estimate_table.index)] = [est1, est2, est3, est4, est5, est6]

    def get_estimates(self):
        """
        Returns the estimate table.
        :return: pandas.dataframe
        """
        return self.__estimate_table

    # - - - estimate functions - - -

    # - - -

    def get_approximate_measure(self):
        return diff_norm(self._f_grid[1:-1, 1:-1], -self.__get_lu_grid(self.__u_grid)[1:-1, 1:-1])

    def get_discrepancy_norm(self):
        return diff_norm(self._f_grid[1:-1, 1:-1], -self.__get_lu_grid(self.__first_approximation)[1:-1, 1:-1])

    def get_exact_solution(self):
        return self.__u_grid

    def get_number_of_iterations(self):
        return self.__total_iterations

    def get_spectral_radius(self):
        return self.__spectral_radius

    def get_min_eigenvalue(self):
        return self.__min_eigenvalue

    def get_max_eigenvalue(self):
        return self.__max_eigenvalue


class SimpleIterationMethod(IterativeMethods):

    def __simple_iteration_approximation(self, i, j):
        p_values = self._get_p_values(i, j)
        q_values = self._get_q_values(i, j)
        return (p_values[0] * self._last_approximation[i - 1, j] / self._x_step_square +
                p_values[1] * self._last_approximation[i + 1, j] / self._x_step_square +
                q_values[0] * self._last_approximation[i, j - 1] / self._y_step_square +
                q_values[1] * self._last_approximation[i, j + 1] / self._y_step_square +
                self._f_grid[i, j]) / \
               (p_values[0] / self._x_step_square + p_values[1] / self._x_step_square +
                q_values[0] / self._y_step_square + q_values[1] / self._y_step_square)

    def solve(self):
        super()._solve(self.__simple_iteration_approximation)


class OptimalParameterMethod(IterativeMethods):

    def __optimal_parameter_approximation(self, i, j, optimal_parameter):
        p_values = self._get_p_values(i, j)
        q_values = self._get_q_values(i, j)
        return self._last_approximation[i, j] + optimal_parameter * (
                p_values[1] * (self._last_approximation[i + 1, j] - self._last_approximation[i, j]) / self._x_step_square -
                p_values[0] * (self._last_approximation[i, j] - self._last_approximation[i - 1, j]) / self._x_step_square +
                q_values[1] * (self._last_approximation[i, j + 1] - self._last_approximation[i, j]) / self._y_step_square -
                q_values[0] * (self._last_approximation[i, j] - self._last_approximation[i, j - 1]) / self._y_step_square +
                self._f_grid[i, j])

    def solve(self):
        super()._solve(self.__optimal_parameter_approximation, self.get_optimal_parameter())

    def get_optimal_parameter(self):
        """
        Returns the optimal method parameter.
        :return: numpy.float64
        """
        return 2 / (self.get_min_eigenvalue() + self.get_max_eigenvalue())


class SeidelMethod(IterativeMethods):

    def __seidel_approximation(self, i, j):
        p_values = self._get_p_values(i, j)
        q_values = self._get_q_values(i, j)
        return ((p_values[0] * self._current_approximation[i - 1, j] / self._x_step_square +
                 p_values[1] * self._last_approximation[i + 1, j] / self._x_step_square +
                 q_values[0] * self._current_approximation[i, j - 1] / self._y_step_square +
                 q_values[1] * self._last_approximation[i, j + 1] / self._y_step_square + self._f_grid[i, j]) /
                (p_values[0] / self._x_step_square + p_values[1] / self._x_step_square +
                 q_values[0] / self._y_step_square + q_values[1] / self._y_step_square))

    def solve(self):
        super()._solve(self.__seidel_approximation)


class UpperRelaxationMethod(IterativeMethods):

    def __upper_relaxation_approximation(self, i, j, optimal_parameter):
        p_values = self._get_p_values(i, j)
        q_values = self._get_q_values(i, j)
        return self._last_approximation[i, j] + optimal_parameter * (
                (self._f_grid[i, j] +
                 p_values[1] * (self._last_approximation[i + 1, j] - self._last_approximation[i, j]) / self._x_step_square -
                 p_values[0] * (self._last_approximation[i, j] - self._current_approximation[i - 1, j]) / self._x_step_square +
                 q_values[1] * (self._last_approximation[i, j + 1] - self._last_approximation[i, j]) / self._y_step_square -
                 q_values[0] * (self._last_approximation[i, j] - self._current_approximation[i, j - 1]) / self._y_step_square) /
                (p_values[0] / self._x_step_square + p_values[1] / self._x_step_square +
                 q_values[0] / self._y_step_square + q_values[1] / self._y_step_square))

    def solve(self):
        super()._solve(self.__upper_relaxation_approximation, self.get_optimal_parameter())

    def get_optimal_parameter(self):
        """
        Returns the optimal method parameter.
        :return: numpy.float64
        """
        return 2 / (1 + np.sqrt(1 - self.get_spectral_radius() ** 2))
