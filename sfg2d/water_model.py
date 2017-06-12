"""Symbolic Calculation of the Water System."""

from sympy import symbols, exp, Matrix, summation, Idx
from sympy.stats import Normal

# Define Symbols
t = symbols('t', real=True)
x = symbols('x', real=True)
t0, t1, t2, t3 = symbols('t:4', positive=True, real=True, nonzero=True)
G = symbols("G", positive=True, real=True, nonzero=True)
s = symbols("s", real=True, positive=True, nonzero=True)
i = symbols("i", integer=True)

# Jacobean der DGL
J = Matrix([
    [-s * G, s * G, 0, 0],
    [s * G, -s * G - 1/t1, 0, 0],
    [0, 1 / t1, -1 / t2, 0],
    [0, 0, 1 / t2, 0]
])

# Eigenwerte
lambdas = J.eigenvals()

# List of Eigenvektors
vs = J.eigenvects()

# Solution of the DGL
solutions = summation(Matrix(vs[int(i)][2]) * exp(vs[int(i)][0]), (i, 0, 4))
solutions = summation([Matrix(vs[i][2]) for i in range(len(vs))])
