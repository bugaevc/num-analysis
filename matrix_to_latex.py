#! /usr/bin/python3

import sys
print(r"\begin{pmatrix}")
for line in sys.stdin.readlines():
    print(*line.split(), sep=' & ', end=r' \\' + '\n')
print(r"\end{pmatrix}")
