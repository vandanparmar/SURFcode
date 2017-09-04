import sys
import os.path
sys.path.append(	
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import dct
from scipy import linalg
import numpy as np


n = 10
no = 4
nu = n
T = 7

d=3


# uns = np.array([[0.448,-0.971,0,0,0],[-0.396,-0.2425,0.6040,0,0],[0,0.8397,-0.1613,0.1466,0],[0,0,0.2864,-0.2316,-0.83339],[0,0,0,0.35447,-0.2251]])
network = dct.network(dct.chain(n))
sim = network.generate_disc_sim()
print(sim.is_controllable())
# sim = dct.disc(n,no=no,nu=nu)
# sim.A = uns
# sim.A = dct.marg_stab(n)

# sim.A /= (np.max(np.abs(linalg.eigvals(sim.A)))+0.01)
sim.A /= 10
#sim.B = np.eye(nu)
sim.C1 = np.eye(2*n)
sim.D12 = np.eye(nu)

# print(sim.A)
# print(sim.B)
# print(sim.C)
print(sim.is_stable())
x0 = np.zeros((2*n,1))
#x0[15] = 1.0
#x0[14] = 1.0
x0[9] = 1.0
#x0[20] = 5.0
#x0[27] = 1.0
sim.setx0(x0)
k = np.arange(0,21)

dct.plot_hmap(sim,k,sim.get_x_set(k),"Uncontrolled Power Network","k")
sim.h2(T,C1=sim.C1,D12=sim.D12,ks=[0,20],heatmap=True)
x = sim.sls_slow(T,d,C1=sim.C1,D12=sim.D12,ks=[0,20],heatmap=True)
# print(x)

#print(R,M)
# print(test_1.is_stable())
# print(test_1.is_controllable())
# test_1.plot_comp(2)
# # # A = np.array([[-1,0],[0,-1]])
# # # B = np.array([[1,0],[1,1],[1,0],[1,0]]).T
# # # C = np.array([[1,0],[0,1]])
# # # #test_1.setC(C)
# # # #test_1.setABC(A,C=C,B=B)
# # # print(test_1.A)
# # # print("A eigvals")
# # # print(linalg.eigvals(test_1.A))
# # # print(test_1.B)
# # # print(test_1.C)
# # # test_1.lqr(R=np.eye(n_b),Q = np.matmul(test_1.C.T,test_1.C),Q_f = np.eye(n_a)*1e6, hor = 10, ks=[1,10],grid=True)
# test_1.inf_lqr(R=np.eye(n_b),Q = np.matmul(test_1.C.T,test_1.C),times=[0,10],grid=True)
# test_1.plot_impulse([0,10],grid=True)
# # # print(test_1.impulse(5))

# print(linalg.det(np.matmul(B,B.conj().T)))
# print(linalg.eigvals(np.matmul(C,C.conj().T)))
# cont, x_c = test_1.is_controllable()
# print(cont)
# print(linalg.eigvals(x_c))
# obvs, y_o = test_1.is_observable()
# print(obvs)
# print(linalg.eigvals(y_o))
# stab, eigs = test_1.is_stable()
# print(stab)
# print(eigs)
# x = test_1.get_x(5)
# y = test_1.get_y(5)

# test_1.plot([500,510])

# A = [[-0.1,0.3],[0.4,-0.2]]
# C = [[1,1],[4,5]]

# test = st.simulate_cont(2,2,2)
# test.setA(A).setC(C)
# test.plot([1,10],grid=True)

# test = st.simulate_disc(2,2,2)
# test.setA(A).setC(C)
# test.plot([1,10],grid=True)

# n = 5
# no = 2
# nu = 3


# pn = dct.network(dct.load_from_file("test/test_graph_1.json"))
# p_cont = pn.generate_disc_sim()
# print(p_cont.is_controllable())
# p_cont.plot_comp()
# p_cont = dct.disc(n = n	,no=no,nu=nu)
# pn.show_network()
# print(np.shape(p_cont.A))
# print(np.shape(p_cont.B))
# print(np.shape(p_cont.C))
# print(linalg.eigvals(p_cont.A))
# ob,w_o = p_cont.is_observable()
# print(ob,linalg.eigvals(w_o))
# co,w_c = p_cont.is_controllable()
# print(co,linalg.eigvals(w_c))
# p_cont.plot_impulse([0,15],grid=True)
# R = np.eye(5)+0.1*dct.random_mat(5,5)
# p_cont.inf_lqr(None,None,ks=[0,20],grid=True)

# p_cont.plot([0,50])
# p_disc.plot([0,50])
















# A = np.array([[0,1],[-5,-2]])
# B = np.array([[0,5],[3,2]])
# C = np.array([[1,0],[0,1]])
# Q = np.array([[1.,0],[0,1]])
# R = np.array([[1,0],[0,1.1]])
# test_cont = dct.disc().setABC(A,B=B,C=C)
# print(test_cont.inf_lqr(R,Q,[1,10]))
