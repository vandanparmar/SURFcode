import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import sim_tools as st
from scipy import integrate
import numpy as np
# test_1 = st.simulate_cont(5,5)
# A = np.array([[-1,0],[0,-1]])
# B = np.array([[1,0],[1,0],[1,0],[1,0]])
# C = np.array([[1,0],[0,1]])
# #test_1.setABC(A,C,B)

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

pn = st.power_network(st.load_from_file("test/test_graph_1.json"))
p_cont = pn.generate_cont_sim()
p_disc = pn.generate_disc_sim()

print(p_cont.A)
print(p_cont.B)

p_cont.plot([0,50])
p_disc.plot([0,50])
