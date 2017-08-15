import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from matplotlib import pyplot as plt
import numpy as np
import dct
from cvxpy import *

n = 6
no = 4
nu = 3
T = 10
R_struc = 1-dct.chain(n)
M_struc = 1-dct.chain(n)
M_struc = M_struc[:nu,:]
print(R_struc)
# network = dct.network(dct.chain(no))
sim = dct.disc(n,no=no,nu=nu)
sim.A = np.multiply(sim.A,dct.chain(n))
#sim.B = np.multiply(sim.B,(1-M_struc).T)
print(sim.A)
print(sim.is_stable())
print(sim.is_controllable())
print(sim.is_observable())
# sim = dct.disc(n=n,no=no,nu=nu)
# sim.setA(dct.chain(n))
# print(np.shape(sim.B))
# sim.setB(np.eye(nu))
# sim.setC(np.eye(no))
uns = np.array([[0.448,-0.971,0,0,0],[-0.396,-0.2425,0.6040,0,0],[0,0.8397,-0.1613,0.1466,0],[0,0,0.2864,-0.2316,-0.83339],[0,0,0,0.35447,-0.2251]])



R = [Variable(n, n) for t in range(0,T)]
M = [Variable(nu,n) for t in range(0,T)]

def H2(R,M,C,D,T):
	toReturn = 0
	for t in range(0,T):
		this = norm(C*R[t]+D*M[t],"fro")
		toReturn += this
	return toReturn

cost = H2(R,M,sim.C,np.random.rand(no,nu),T)
constr = [R[0]==np.eye(n)]
for t in range(0,T-1):
	# constr += [mul_elemwise(R_struc,R[t])==0]
	# constr += [mul_elemwise(M_struc,M[t])==0]
	constr += [R[t+1] == sim.A*R[t]+sim.B*M[t]]
constr += [R[T-1]==0]
prob = Problem(Minimize(cost),constr)
a = prob.solve()
print(a)
R = np.array(list(map(lambda x: x.value,R)))
M = np.array(list(map(lambda x: x.value,M))) 

t_plot = 20

sim.setx0(np.array([[0,1,0,0,0,0]]).T)
x = sim.x0
delta_x = np.zeros((n,T))
xhat = np.zeros((n,1))
u = np.zeros((nu,1))
for t in range(0,t_plot-1):
	delta_x = np.append(delta_x,np.array([x[:,t]-xhat[:,t]]).T,axis=1)
	for i in range(0,T):
		u[:,t]+= np.matmul(M[i],delta_x[:,t-i+T])
		xhat[:,t] += np.matmul(R[i],delta_x[:,t-i+T])
	w = np.zeros((n,))
#	w = np.matmul(sim.B,np.random.normal(0,0.1,nu))
	x_plus1 = np.array([np.matmul(sim.A,x[:,-1])+np.matmul(sim.B,u[:,-1])+w]).T
	x = np.append(x,x_plus1,axis=1)
	u = np.append(u,np.zeros((nu,1)),axis=1)
	xhat = np.append(xhat,np.zeros((n,1)),axis=1)


times = np.arange(0,t_plot)
plt.pcolor(np.absolute(x), cmap="BuPu")
plt.show()

sim.get_x_set([0,t_plot])
dct.plot_sio(sim,times,True,True,x=x,u=u)
dct.plot_sio(sim,times,True,True,x=sim.get_x_set(times))