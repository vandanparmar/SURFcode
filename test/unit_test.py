import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import sim_tools as st

#continuous
print('Continuous')

test_cont = st.simulate_cont()
test_cont = st.simulate_cont(3)
print('Expect None, None.')
print(test_cont.B,test_cont.C)
test_cont = st.simulate_cont(3,4)
test_cont = st.simulate_cont(3,4,5)

A = [[-0.1,0.8],[0,-0.1]]
B = [0.4,0.3]
C = [[0.2],[0.4]]
x0 = [1,1]

print('SetABC')
test_cont = st.simulate_cont().setABC(A,B,C)
test_cont.setA(A)
test_cont.setB(B)
test_cont.setC(C)

test_cont.setx0(x0)
test_cont.set_plot_points(30)
print("X,Y with grid.")
test_cont.plot([1,100],grid=True)
print("X,Y no grid.")
test_cont.plot([1,100],grid=False)
print("Y with grid.")
test_cont.plot_output([1,100],grid=True)
print("Y no grid.")
test_cont.plot_output([1,100],grid=False)

test_cont.C = None
print("X,Y no C, with grid.")
test_cont.plot([1,100],grid=True)
print("X,Y no C, no grid.")
test_cont.plot([1,100],grid=False)

#discrete
print('Discrete')


test_disc = st.simulate_disc()
test_disc = st.simulate_disc(3)
print('Expect None, None.')
print(test_cont.B,test_disc.C)
test_disc = st.simulate_disc(3,4)
test_disc = st.simulate_disc(3,4,5)

print('SetABC')
test_disc = st.simulate_disc().setABC(A,B,C)
test_disc.setA(A)
test_disc.setB(B)
test_disc.setC(C)

test_disc.setx0(x0)
print("X,Y with grid.")
test_disc.plot([1,100],grid=True)
print("X,Y no grid.")
test_disc.plot([1,100],grid=False)
print("Y with grid.")
test_disc.plot_output([1,100],grid=True)
print("Y no grid.")
test_disc.plot_output([1,100],grid=False)

test_disc.C = None
print("X,Y no C, with grid.")
test_disc.plot([1,100],grid=True)
print("X,Y no C, no grid.")
test_disc.plot([1,100],grid=False)
