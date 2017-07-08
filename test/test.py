import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import sim_tools as st

# test_1 = simulate_cont(5,5)
# A = np.array([[-1,0],[0,-1]])
# B = np.array([[1,0],[1,0],[1,0],[1,0]])
# C = np.array([[1,0],[0,1]])
# #test_1.setABC(A,C,B)

# x = test_1.get_x(5)
# y = test_1.get_y(5)

# test_1.plot([500,510])

C = []

test = st.simulate_disc(2,2,2)
print(test.C)
test.plot([1,10],grid=True)