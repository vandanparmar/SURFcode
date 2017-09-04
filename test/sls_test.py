import sys
import os.path
sys.path.append(	
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import dct
from scipy import linalg
import numpy as np
import time
import json
from matplotlib import pyplot as plt
n = 100
no = 1
nu = n
T = 9

d=2


sim = dct.disc(n,no=no,nu=nu)
sim.A = dct.marg_stab(n)
sim.B = np.eye(nu)
sim.C1 = np.eye(n)
sim.D12 = np.eye(nu)

ns = np.arange(1,6)
ns = ns**2
ns *= 4
# ns = [50,100,150,200]
print(ns)

#times = {'sls':[],'sls_fast':[],'h2':[],'ns':ns.tolist()}
times = {'sls':[],'h2':[],'sls_slow':[],'sls_fast':[],'ns':ns.tolist()}

for n in ns:
	print(n)
	sim = dct.disc(n,no=no,nu=n)
	sim.A = dct.marg_stab(n)
	sim.B = np.eye(n)
	sim.C1 = np.eye(n)
	sim.D12 = np.eye(n)
	h2_i = 0
	sls_fast_i = 0
	sls_i = 0
	sls_slow_i = 0
	for i in range(0,3):
		t0 = time.time()
		[x,y,h2] = sim.h2(T,sim.C1,sim.D12)
		t1 = time.time()
		[x,y,sls] = sim.sls(T,2,sim.C1,sim.D12)
		t2 = time.time()
		[x,y,sls_fast] = sim.sls_fast(T,2,sim.C1,sim.D12)
		[x,y,sls_slow] = sim.sls_slow(T,2,sim.C1,sim.D12)
		t3 = time.time()
		h2_i += h2
		sls_fast_i += sls_fast
		sls_i += sls
		sls_slow_i += sls_slow
	times['sls'].append(sls_i/3)
	times['h2'].append(h2_i/3)
	times['sls_fast'].append(sls_fast_i/3)
	times['sls_slow'].append(sls_slow_i/3)

	with open('test/data_fast_proper_all.json', 'w') as outfile:
		json.dump(times, outfile)
	print(times)
	print("done: " + str(n))

