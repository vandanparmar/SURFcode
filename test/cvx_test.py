# Generate data for control problem.
# import numpy as np
# np.random.seed(1)
# n = 8
# m = 2
# T = 50
# alpha = 0.2
# beta = 5
# A = np.eye(n) + alpha*np.random.randn(n,n)
# B = np.random.randn(n,m)
# x_0 = beta*np.random.randn(n,1)


# from cvxpy import *
# x = Variable(n, T+1)
# u = Variable(m, T)

# states = []
# for t in range(T):
#     cost = sum_squares(x[:,t+1]) + sum_squares(u[:,t])
#     constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
#               norm(u[:,t], 'inf') <= 1]
#     states.append( Problem(Minimize(cost), constr) )
# # sums problem objectives and concatenates constraints.
# prob = sum(states)
# prob.constraints += [x[:,T] == 0, x[:,0] == x_0]
# a = prob.solve()
# print(a)

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import numpy as np
import dct
from cvxpy import *
# n = 2
# alpha = 0.2


# A = -np.eye(2)
# #A = -np.eye(n) - alpha*np.random.randn(n,n)
# print(A)
# P = Variable(n,n)

# cost = 0
# constr = [A.T*P+P*A<=0, P>=0, trace(P)==1]
# prob = Problem(Minimize(cost),constr)
# a = prob.solve()
# print(a)
# print(P.value)


n = 4
no = 3
nu = 4
T = 10
sim = dct.disc(n=n,no=no,nu=nu)

R = [Variable(n, n) for t in range(0,T)]
M = [Variable(nu,n) for t in range(0,T)]

def H2(R,M,C,D,T):
	toReturn = 0
	for t in range(0,T):
		this = norm(C*R[t]+D*M[t])
		toReturn += this
	return toReturn

cost = H2(R,M,sim.C,np.random.rand(no,nu),T)
constr = [R[0]==np.eye(n)]
for t in range(1,T-1):
	constr += [R[t+1] == sim.A*R[t]+sim.B*M[t]]
constr += [R[T-1]==np.zeros((n,n))]
prob = Problem(Minimize(cost),constr)
a = prob.solve()
print(a)
print(R.value)
# phi = Variable(n+nu,nu)

# C1 = sim.C
# D21 = np.random.rand(no,nu)
# print(np.shape(C1))
# print(np.shape(D21))
# Cs = np.reshape(np.append(C1,D21),(4,8))
# A = np.eye(n)-sim.A
# B2 = sim.B
# print(np.shape(A))
# print(np.shape(B2))
# As = np.reshape(np.append(A,B2),(5,8))
# I_n_nu = np.zeros((n,nu))
# for i in range(0,nu):
# 	I_n_nu[i,i]=1

# cost = norm(Cs*phi)
# constr = [As*phi == I_n_nu]
# prob = Problem(Minimize(cost),constr)
# a = prob.solve()
# print(a)
# print(phi.value)
