#! /usr/bin/python3

import math
import argparse
import itertools

def runge_kutta(a, b, n, fs, y0s, order=4):
    assert order in (2, 4)
    h = (b - a) / n
    x = a
    ys = y0s
    yield x, ys
    for i in range(n):
        k1s = [f(x, *ys) for f in fs]
        k2s = [f(x + h/2, *(y + h/2 * k1 for y, k1 in zip(ys, k1s))) for f in fs]
        if order == 4:
            k3s = [f(x + h/2, *(y + h/2 * k2 for y, k2 in zip(ys, k2s)))
                  for f in fs]
            k4s = [f(x + h, *(y + h * k3 for y, k3 in zip(ys, k3s))) for f in fs]
            ys = [y + (k1 + 2*k2 + 2*k3 + k4) * h / 6
                  for y, k1, k2, k3, k4 in zip(ys, k1s, k2s, k3s, k4s)]
        else:
            ys = [y + k2 * h for y, k2 in zip(ys, k2s)]
        x += h
        yield x, ys


def intersect_lines(a1, b1, c1, a2, b2, c2):
    """Intersect two lines:
    a1*x + b1*y = c1
    a2*x + b2*y = c2
    """
    x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
    y = (c1*a2 - c2*a1) / (b1*a2 - b2*a1)
    return x, y

def thomas(A, B, C, F, first, last):
    '''Solve a system of linear equations

    A[i]*x[i-1] + C[i]*x[i] + B[i]*x[i+1] = F[i] for i in range(n-1)
    first[0]*x[-1] + first[1]*x[0] = first[2]  # x[-1] is NOT the last one
    last[0]*x[n-2] + last[1]*x[n-1] = last[2]
    Return all the values from x[-1] up to (and including) x[n-1]
    '''
    n = len(F) + 1
    alpha = [-first[1]]
    beta = [first[2]]
    alpha[0] /= first[0]
    beta[0] /= first[0]
    for i in range(n-1):
        # alpha&beta[i+1]
        alpha.append(-B[i])
        beta.append(F[i] - A[i]*beta[i])  
        alpha[-1] /= A[i]*alpha[i] + C[i]
        beta[-1] /= A[i]*alpha[i] + C[i]
    x = [None] * n
    x[n-1] = intersect_lines(1, -alpha[n-1], beta[n-1], *last)[1]
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return [alpha[0] * x[0] + beta[0]] + x


def boundary_value_problem(a, b, n, p, q, r, sa, ga, da, sb, gb, db):
    '''Solve differential linear equation y'' + p(x)y' + q(x)y + r(x) = 0.

    The equation is solved using the finite difference method.
    The segment [a, b] is divided into n parts.
    sa*y(a) + ga*y'(a) = da,
    sb*y(b) + gb*y'(b) = db
    are boundary values.
    '''
    h = (b - a) / n
    A = []
    B = []
    C = []
    F = []
    for i in range(1, n):
        x = a + h * i
        A.append(1/h**2 - p(x)/2/h)
        B.append(1/h**2 + p(x)/2/h)
        C.append(q(x) - 2/h**2)
        F.append(r(x))
    first = [sa - ga/h, ga/h, da]
    last = [-gb/h, sb + gb/h, db]
    for i, y in zip(range(n+1), thomas(A, B, C, F, first, last)):
        x = a + h * i
        yield x, y

def build_globals():
    """Build a namespace that's suitable to be used in eval()."""
    import builtins
    res = dict()
    for module in (builtins, math):
        res.update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})
    return res

def ask_for(prompt, quiet=False):
    if quiet:
        return input()
    return input(prompt + ': ')

def output(*args, trunc=True):
    if not trunc:
        print(*args)
    else:
        print(*('{:f}'.format(arg) for arg in args))

def make_function(s, args, gl=build_globals()):
    f = compile(s, 'stdin', 'eval')
    return lambda *a: eval(f, gl, {name: val for name, val in zip(args, a)})

def main():
    parser = argparse.ArgumentParser(description='Solve differential equations.')
    parser.add_argument('method', choices=('runge-kutta', 'bound', 'sample'),
                        help='the method to use for solving equations')
    parser.add_argument('a', type=float, help='left bound')
    parser.add_argument('b', type=float, help='right bound')
    parser.add_argument('n', type=int, help='number of iterations')
    parser.add_argument('-2', '--order-2', action='store_true',
                        help='use runge-kutta algorithm of second order')
    parser.add_argument('-n', '--no-trunc', action='store_true',
                        help='do not truncate output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='do not display prompts')

    args = parser.parse_args()
    trunc = not args.no_trunc
    if args.method == 'runge-kutta':
        order = 2 if args.order_2 else 4
        cnt = int(ask_for('Number of functions', args.quiet))
        fs = []
        y0s = []
        for i in range(cnt):
            y0 = float(ask_for('Initial value #{}'.format(i+1), args.quiet))
            y0s.append(float(y0))
            vars = ['x'] + ['y' + str(i+1) for i in range(cnt)]
            f = ask_for('Function #{}'.format(i+1), args.quiet)
            fs.append(make_function(f, vars))
        for x, ys in runge_kutta(args.a, args.b, args.n, fs, y0s, order):
            output(x, *ys, trunc=trunc)
    elif args.method == 'sample':
        cnt = int(ask_for('Number of functions', args.quiet))
        fs = []
        for i in range(cnt):
            f = ask_for('Function #{}'.format(i+1), args.quiet)
            fs.append(make_function(f, ['x']))
        h = (args.b - args.a) / args.n
        x = args.a
        for i in range(args.n):
            output(x, *[f(x) for f in fs], trunc=trunc)
            x += h
        output(x, *[f(x) for f in fs], trunc=trunc)
    elif args.method == 'bound':
        p = make_function(ask_for('p', args.quiet), ['x'])
        q = make_function(ask_for('q', args.quiet), ['x'])
        r = make_function(ask_for('r', args.quiet), ['x'])
        sa = float(ask_for('sigma_a', args.quiet))
        ga = float(ask_for('gamma_a', args.quiet))
        da = float(ask_for('delta_a', args.quiet))
        sb = float(ask_for('sigma_b', args.quiet))
        gb = float(ask_for('gamma_b', args.quiet))
        db = float(ask_for('delta_b', args.quiet))
        for x, y in boundary_value_problem(args.a, args.b, args.n, p, q, r, sa, ga, da, sb, gb, db):
            output(x, y, trunc=trunc)

if __name__ == '__main__':
    main()