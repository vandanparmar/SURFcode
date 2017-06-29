import numpy as np
from scipy.integrate import ode
from scipy.linalg import expm
import matplotlib.pyplot as plt


#analytic function of open loop matrix differential equation
def analytic(x0,A,t):
	x = np.matmul(expm(A*t),x0)
	return x

#plotter for solution of open loop matrix differential equation
def plot_analytic(analytic_f,dt,tot_t):
	t = np.arange(0,tot_t,dt)
	x = np.zeros((len(t),4))
	for i,time in enumerate(t):
		x[i,:] = analytic(x0,A,time)
	plt.plot(t,x)
	plt.show()




x0 = np.array([1,1,0.5,0.3])
A = np.matrix([[-0.5,0,0,0],[0,-0.3,0,0],[-1,-1,-1,-1],[0.1,0.1,0.1,0.1]])
tot_t = 10
dt = 0.1

plot_analytic(analytic,0.1,10)


