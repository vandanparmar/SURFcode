import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from matplotlib import pyplot as plt
import numpy as np
import dct
from cvxpy import *
from matplotlib import colors,cm
import multiprocessing as mp
from itertools import repeat

def mar_stab(n):
	"""Generates a chain graph structure with n nodes.

	Args:
		n (int): Number of nodes

	Returns:
		ndarray: Adjacency matrix 
	"""	
	def p(i,j):
		if(i==j):
			return 0.6
		elif(i==j+1):
			return 0.2
		elif(i==j-1):
			return 0.2
		else:
			return 0
	toReturn = [[p(i,j) for i in range(0,n)] for j in range(0,n)]
	return np.array(toReturn)

def binify(A):
	toReturn = np.array((A!=0.0))
	return toReturn

n = 31
no = 20
nu = n
T = 10	

d=2

# uns = np.array([[0.448,-0.971,0,0,0],[-0.396,-0.2425,0.6040,0,0],[0,0.8397,-0.1613,0.1466,0],[0,0,0.2864,-0.2316,-0.83339],[0,0,0,0.35447,-0.2251]])
sim = dct.disc(n,no=no,nu=nu)
# sim.A = uns
A = dct.marg_stab(n)
A[0,0] = 1.2
A[n-1,n-1] = 1.2
sim.A = A
# sim.A = np.multiply(sim.A,dct.chain(n))
sim.B = np.eye(nu)
sim.C1 = np.eye(n)
sim.D12 = np.eye(nu)
#sim.B = np.multiply(sim.B,(1-M_struc).T)

# R = np.array([Variable(n, n) for t in range(0,T)])
# M = np.array([Variable(nu,n) for t in range(0,T)])


# R_struc = [np.linalg.matrix_power(binify(sim.A),d-1) for t in range(0,T)]
# M_struc = [binify(np.matmul(sim.B.T,np.array(R_struc[t]))) for t in range(0,T)]
# R_struc = np.swapaxes(np.array(R_struc),0,2).tolist()
# M_struc = np.swapaxes(np.array(M_struc),0,2).tolist()
# R_struc = np.swapaxes(np.array(R_struc),1,2).tolist()
# M_struc = np.swapaxes(np.array(M_struc),1,2).tolist()

# print(np.shape(R_struc))
# print(np.shape(M_struc))

# R = list(map(lambda i: (list(map(lambda j: vstack(list(map(lambda k: Variable() if k else 0, j))), i))),R_struc))
# M = list(map(lambda i: (list(map(lambda j: vstack(list(map(lambda k: Variable() if k else 0, j))), i))),M_struc))

# def H2(R,M,C,D,T):
# 	toReturn = 0
# 	for t in range(0,T):
# 		vec1 = C*R[t]
# 		vec2 = D*M[t]
# 		toReturn += norm(vec1)
# 		toReturn += norm(vec2)
# 	return toReturn

# def eval_i(A,B,R,M,C1,D12,T,eye):
# 	eye = eye.T
# 	# print([R[T-1]==0])
# 	cost = H2(R,M,C1,D12,T)
# 	# print(cost)
# 	constr = [R[0]==eye]
# 	for t in range(0,T-1):
# 		constr += [R[t+1]==A*R[t]+B*M[t]]
# 	constr += [R[T-1]==0]
# #	print(constr)
# 	prob = Problem(Minimize(cost),constr)
# 	a = prob.solve()
# 	print(a)
# 	R = np.array(list(map(lambda x: x.value,R)))
# 	M = np.array(list(map(lambda x: x.value,M)))
# 	return (R,M)


# pool = mp.Pool(processes=mp.cpu_count())

# args_list = zip(repeat(sim.A),repeat(sim.B),R,M,repeat(sim.C1),repeat(sim.D12),repeat(T),np.eye(n))

# res = pool.starmap(eval_i,args_list)

# [R,M] = list(zip(*res))

# #i=0
# # for i in range(0,n):
# # 	R_i = R[i]
# # 	M_i = M[i]
# # 	eye = np.eye(n)[i,:]
# # 	[R[i],M[i]] = eval_i(sim.A,sim.B,R_i,M_i,sim.C1,sim.D12,T,eye)

# print(np.shape(R))

# R = np.swapaxes(np.array(R)[:,:,:,0],1,2)
# M = np.swapaxes(np.array(M)[:,:,:,0],1,2)
# R = np.swapaxes(R,0,2)
# M = np.swapaxes(M,0,2)
# # print(R)
# # print(M)
# # cost = H2(R,M,sim.C1,sim.D12,T)
# # constr = [R[0]==np.eye(n)]
# # for t in range(0,T-1):
# # 	constr += [R[t+1] == sim.A*R[t]+sim.B*M[t]]
# # constr += [R[T-1]==0]
# # prob = Problem(Minimize(cost),constr)
# # #a = prob.solve(solver=SCS,eps = 1e-10)
# # a = prob.solve()
# # print(a)
# # R = np.array(list(map(lambda x: x.value,R)))
# # M = np.array(list(map(lambda x: x.value,M))) 




# t_plot = 20

x0 = np.zeros((n,1))
# #x0[15] = 1.0
x0[15] = 10.0
# x0[15] = 10.0
# #x0[21] = 5.0
# #x0[27] = 1.0
sim.setx0(x0)
# x = sim.x0
# delta_x = np.zeros((n,T))
# xhat = np.zeros((n,1))
# u = np.zeros((nu,1))
# for t in range(0,t_plot-1):
# 	delta_x = np.append(delta_x,np.array([x[:,t]-xhat[:,t]]).T,axis=1)
# 	for i in range(0,T):
# 		u[:,t]+= np.matmul(M[i],delta_x[:,t-i+T])
# 		xhat[:,t] += np.matmul(R[i],delta_x[:,t-i+T])
# 	w = np.zeros((n,))
# #	w = np.matmul(sim.B,np.random.normal(0,0.1,nu))
# 	x_plus1 = np.array([np.matmul(sim.A,x[:,-1])+np.matmul(sim.B,u[:,-1])+w]).T
# 	x = np.append(x,x_plus1,axis=1)
# 	u = np.append(u,np.zeros((nu,1)),axis=1)
# 	xhat = np.append(xhat,np.zeros((n,1)),axis=1)
sim.h2(5,ks=[0,20],heatmap=True)
sim.h2(6,ks=[0,20],heatmap=True)
sim.sls_slow(6,2,ks=[0,20],heatmap=True)
my_cmap = cm.get_cmap('BuPu')
my_cmap.set_bad((0.9686275,0.9882359411,0.9921568627))

vmin = 1e-7
times = np.arange(0,t_plot)
# x_free = sim.get_x_set(times)
# plt.title("State Uncontrolled")
# plt.pcolor(np.absolute(x_free), norm=colors.LogNorm(), cmap=my_cmap,vmin=vmin)
# plt.colorbar()
# plt.xlabel('k')
# plt.show()

plt.title("State Controlled")
plt.pcolor(np.absolute(x), norm=colors.LogNorm(), cmap=my_cmap,vmin=vmin)
plt.colorbar()
plt.show()

plt.pcolor(np.absolute(u),norm=colors.LogNorm(),cmap=my_cmap,vmin=vmin)
plt.title("Input")
plt.colorbar()
plt.show()


# dct.plot_sio(sim,times,True,True,x=x,u=u)
# dct.plot_sio(sim,times,True,True,x=x_free)
