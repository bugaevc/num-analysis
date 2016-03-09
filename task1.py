#! /usr/bin/python3.5

import sys
import math
from copy import deepcopy
import argparse
from fractions import Fraction
from decimal import Decimal


def gauss(matrix, smart=True):
    """Solve a system of linear equations

    Transform matrix so that its first NxN part is the identity matrix.
    This effectively solves the system of equations for all given right columns.
    Return the determinant of the initial matrix.
    """
    n = len(matrix)
    det = 1  # invariant: det * |matrix| = |initial matrix|
    column_swaps = []

    def swap_lines(i, j):
        nonlocal det
        if i == j:
            return
        det *= -1
        matrix[i], matrix[j] = matrix[j], matrix[i]

    def swap_columns(i, j):
        nonlocal det
        if i == j:
            return
        det *= -1
        for line in matrix:
            line[i], line[j] = line[j], line[i]

    def normalize_line(i):
        nonlocal det
        det *= matrix[i][i]
        k = 1 / matrix[i][i]
        matrix[i] = [el * k for el in matrix[i]]

    def sub(i, j):
        """Subtract line[i]*matrix[j][i] from line[j]."""
        matrix[j] = [lj - li * matrix[j][i] for li, lj in zip(matrix[i], matrix[j])]

    for i in range(n):
        if smart:
            j = max(range(n), key=lambda j: abs(matrix[i][j]))
            column_swaps.append((i, j))
            swap_columns(i, j)
        else:
            j = next(j for j in range(i, n) if matrix[j][i])
            swap_lines(i, j)
        normalize_line(i)
        for j in range(i+1, n):
            sub(i, j)
    for i in reversed(range(n)):
        for j in range(i):
            sub(i, j)
    for i, j in reversed(column_swaps):
        swap_columns(i, j)
        swap_lines(i, j)
    return det


def relax(matrix, w, eps=10**-8):
    """Solve a system of linear equations.

    Iterative method of successive over-relaxation.
    Return the vector of answers.
    """
    n = len(matrix)

    def transpose(matrix):
        return list(zip(*matrix))

    def mul(m1, m2):
        """Multiply an NxN matrix by Nx(N+1) matrix.

        All elements beyond that size are simply ignored.
        """
        return [[sum(m1[i][k] * m2[k][j] for k in range(n)) for j in range(n+1)] for i in range(n)]

    # force matrix to be symmetric and positive-definite
    matrix = mul(transpose(matrix), matrix)
    res = [1] * n  # starting values
    max_diff = math.inf
    while max_diff >= eps:
        max_diff = 0
        for i in range(n):
            diff = matrix[i][n] - sum(matrix[i][j] * res[j] for j in range(n))
            max_diff = max(abs(diff), max_diff)
            res[i] += w/matrix[i][i] * diff
    return res

#
# Input/output and utilities
#

def build_globals():
    """Build a namespace that's suitable to be used in eval()."""
    import builtins
    res = dict()
    for module in (builtins, math):
        res.update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})
    res['Fraction'] = Fraction
    res['Decimal'] = Decimal
    return res

def ask_for(prompt, quiet=False):
    if quiet:
        return input()
    return input(prompt + ': ')

def output(trunc=True, *args):
    if not trunc:
        print(*args)
    else:
        print(*('{:f}'.format(arg) for arg in args))

def read_matrix(interactive=False, type_=float, quiet=False):
    """Read a matrix from stdin, either directly or read a formula and use it to generate the matrix."""
    if not interactive:
        return [[type_(el) for el in line.split()] for line in sys.stdin.readlines()]

    n = int(ask_for('n', quiet))
    x = type_(ask_for('x', quiet))
    expr1 = ask_for('element formula', quiet)
    expr2 = ask_for('RHS formula', quiet)

    gl = build_globals()

    return [
        [type_(eval(expr1, gl, {'x': x, 'n': n, 'i': i, 'j': j})) for j in range(n)] +
        [type_(eval(expr2, gl, {'x': x, 'n': n, 'i': i}))]
        for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description='Solve a system of linear equations')
    parser.add_argument('method', choices=('gauss', 'gauss_smart', 'relax'),
                        help='The method to use for solving the system')
    parser.add_argument('-d', '--det', action='store_true',
                        help='Output determinant of the matrix (not supported when the method is "relax")')
    parser.add_argument('-i', '--inv', action='store_true',
                        help='Output the inverse matrix (not supported when the method is "relax")')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Output all the calculated values.')
    parser.add_argument('-n', '--no-trunc', action='store_true',
                        help='Do not truncate output.')
    parser.add_argument('-t', '--type', choices=('float', 'Fraction', 'Decimal'),
                        default='float', help='Python type to use for calculations')
    parser.add_argument('-f', '--formula', action='store_true',
                        help='Use formulas to calculate the matrix instead of inputting it directly')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not output prompts and extra interesting information')

    args = parser.parse_args()
    type_ = {'float': float, 'Fraction': Fraction, 'Decimal': Decimal}[args.type]
    if args.method == 'relax':
        w = type_(ask_for('w', args.quiet))
    matrix = read_matrix(args.formula, type_, args.quiet)
    n = len(matrix)

    output_answer = args.all or not (args.det or args.inv)
    trunc = not args.no_trunc and type_ is not Fraction

    if args.method == 'relax':
        if output_answer:
            # who knows, maybe we have nothing to do
            answer = relax(matrix, w)
    else:
        matrix_copy = deepcopy(matrix)
        if args.inv or args.all:
            # append the identity matrix in order to calculate the inverse matrix
            for i, line in enumerate(matrix_copy):
                line += [i == j for j in range(n)]
        det = gauss(matrix_copy, smart=args.method == 'gauss_smart')
        if args.all and not args.quiet:
            print('det = ', end='')
        if args.all or args.det:
            output(trunc, det)
        if args.all or args.inv:
            if args.all and not args.quiet:
                print('The inverse matrix:')
            for line in matrix_copy:
                output(trunc, *line[n+1:])
        if output_answer:
            answer = [line[n] for line in matrix_copy]
    if output_answer:
        if args.all and not args.quiet:
            print('The answer:')
        output(trunc, *answer)
        if not args.quiet:
            print('Residual:')
            output(trunc, *(line[n] - sum(a*x for a, x in zip(line, answer)) for line in matrix))

if __name__ == '__main__':
    main()
