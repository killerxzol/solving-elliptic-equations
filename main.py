import numpy as np
from src import IterativeMethods


def u_function(x, y):
    return x ** 2 * y ** 2 * (1 + y)


def f_function(x, y):
    return -(2 * y ** 2 + 2 * y ** 3 + 8 * x * y ** 2 + 8 * x * y ** 3 + 2 * x ** 2 + 6 * x ** 2 * y)


def p_function(x, y):
    return 1 + 2 * x


def q_function(x, y):
    return 1


if __name__ == '__main__':

    methods = ['simple-iteration', 'optimal-parameter', 'seidel', 'upper-relaxation']

    constants = [1, 3, 1, 1]        # [c1, c2, d1, d2], where [0 < c1 <= p(x, y) <= c2], [0 < d1 <= q(x, y) <= d2]

    args = [u_function, f_function, p_function, q_function, constants, 1, np.pi, 5, 5, 1e-5]

    i = 1                           # current method

    if i == 0:
        method = IterativeMethods.SimpleIterationMethod(*args)
    elif i == 1:
        method = IterativeMethods.OptimalParameterMethod(*args)
    elif i == 2:
        method = IterativeMethods.SeidelMethod(*args)
    elif i == 3:
        method = IterativeMethods.UpperRelaxationMethod(*args)
    else:
        raise Exception('Unknown method.')

    method.solve()

    # print the program report

    np.set_printoptions(precision=4, suppress=True)

    table = method.get_estimates()
    print(f'Method: {methods[i]}.')
    print(f'\n1. Measure of approximation:  ||F-AU_*|| = {method.get_approximate_measure()}')
    print(f'\n2. Discrepancy norm for U^0:  ||F-AU^0|| = {method.get_discrepancy_norm()}')
    print(f'\n3. Number of iterations:  m = {method.get_number_of_iterations()}')
    print(f'\n4. Spectral radius:  rho(H) = {method.get_spectral_radius()}')

    table.index = table.index + 1
    print(f'\nTable:\n{table.to_markdown(index=True)}')

    print('\nApproximate solution:\n', method.get_solution())
    print('\nExact solution:\n', method.get_exact_solution())
