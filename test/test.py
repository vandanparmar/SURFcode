import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import dct
from scipy import linalg
import numpy as np
n_a = 9
n_b = 7
n_c = 5
test_1 = dct.disc(n_a,no=n_c,nu=n_b)
print(test_1.is_stable())
print(test_1.is_controllable())
test_1.plot_comp(2)
# A = np.array([[-1,0],[0,-1]])
# B = np.array([[1,0],[1,1],[1,0],[1,0]]).T
# C = np.array([[1,0],[0,1]])
# #test_1.setC(C)
# #test_1.setABC(A,C=C,B=B)
# print(test_1.A)
# print("A eigvals")
# print(linalg.eigvals(test_1.A))
# print(test_1.B)
# print(test_1.C)
#test_1.lqr(R=np.eye(n_b),Q = np.matmul(test_1.C.T,test_1.C),Q_f = np.eye(n_a)*1e6, hor = 60, ks=[56,60],grid=True)
#print(test_1.impulse(5))

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

#pn = dct.network(dct.load_from_file("test/test_graph_1.json"))
#p_cont = pn.generate_disc_sim()
#print(p_cont.is_controllable())
#p_cont.plot_comp()
#p_cont = st.simulate_cont(3,4,5)
# pn.show_network()
# print(np.shape(p_cont.A))
# print(np.shape(p_cont.B))
# print(np.shape(p_cont.C))
# print(linalg.eigvals(p_cont.A))
# ob,w_o = p_cont.is_observable()
# print(ob,linalg.eigvals(w_o))
# co,w_c = p_cont.is_controllable()
# print(co,linalg.eigvals(w_c))
#p_cont.plot_step([0,15],grid=True,inputs=[4,5])
#R = np.eye(5)+0.1*dct.random_mat(5,5)
#p_cont.inf_lqr(R,np.matmul(p_cont.C.T,p_cont.C),ks=[0,20],grid=True)

# p_cont.plot([0,50])
# p_disc.plot([0,50])
