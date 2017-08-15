import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from matplotlib import pyplot as plt
import numpy as np
import dct
from cvxpy import *
from scipy import linalg

n = 4
nu = 3
no = 2

sim = dct.cont(n=n,nu=nu,no=no)



P = Semidef(n)
constr = [sim.A.T*P + P*sim.A <= 0, P >= 0, trace(P)==1]

prob = Problem(Minimize(0),constr)
a = prob.solve()
print(a)
print(linalg.eigvals(P.value))
print(linalg.eigvals(sim.A))