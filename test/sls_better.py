import sys
import os.path
sys.path.append(	
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import numpy as np
from scipy import linalg
from dct.tools import *
import dct
import cvxpy
import multiprocessing as mp
from itertools import repeat
import time

n = 50
no = 1
nu = n
T = 7	

d=3




sim = dct.disc(n,no=no,nu=nu)
sim.A = dct.marg_stab(n)
# B = np.array([[0,0,0,0,1,0],[0,0,1,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,0]])
# sim.B = B
sim.B = np.eye(nu)
sim.C1 = np.eye(n)
sim.D12 = sim.B

network = dct.network(dct.chain(n))
# sim = network.generate_disc_sim()
# sim.C1 = np.eye(2*n)
# sim.D12 = sim.B
# print(sim.A,sim.B)

# print(np.shape(sim.B))
# print(np.shape(sim.A))
x0 = np.zeros((n,1))
x0[25] = 10
sim.setx0(x0)

# print(sim.sls_slow(T,d,sim.C1,sim.D12))
# print(sim.sls(T,d,sim.C1,sim.D12))
print(sim.sls_fast(T,d,sim.C1,sim.D12,ks=[0,20],heatmap=True))






