from simulation_tools import *

# test_1 = simulate_cont(5,5)
# A = np.array([[-1,0],[0,-1]])
# B = np.array([[1,0],[1,0],[1,0],[1,0]])
# C = np.array([[1,0],[0,1]])
# #test_1.setABC(A,C,B)

# x = test_1.get_x(5)
# y = test_1.get_y(5)

# test_1.plot([500,510])

test = simulate_disc()
test.plot([1,10])