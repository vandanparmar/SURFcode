import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import sim_tools as st
from scipy import linalg
import numpy as np
test_1 = st.simulate_cont(2,3,4)
A = np.array([[-1,0],[0,-1]])
B = np.array([[1,0],[1,1],[1,0],[1,0]]).T
C = np.array([[1,0],[0,1]])
#test_1.setABC(A,C=C,B=B)
print(test_1.A)
print(test_1.B)
print(test_1.C)
test_1.plot_impulse([0,100],inputs=[1,2],outputs=[2,1],grid=True)


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

# pn = st.power_network(st.load_from_file("test/test_graph_1.json"))
# p_cont = pn.generate_cont_sim().setx0(np.transpose([[10]*10]))
# p_disc = pn.generate_disc_sim()

# print(p_cont.A)
# print(p_cont.B)

# p_cont.plot([0,50])
# p_disc.plot([0,50])
