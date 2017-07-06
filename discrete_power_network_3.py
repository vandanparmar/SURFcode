import numpy as np
from scipy import integrate
from scipy.linalg import expm
import matplotlib.pyplot as plt
from graph_tools import *


#step function for integrator, of the form x' = f(x,t)
def ode_step(t,x,A,B,u):
	xdot = np.matmul(A,x)+np.matmul(B,u)
	return xdot


#simulator, to carry out simulation
def network_sim_integrator(x0,A,B,C,u,dt,totalTime):
	sol = np.array([x0])
	integrator = integrate.ode(ode_step).set_integrator("dop853")
	integrator.set_initial_value(x0,0.0)
	integrator.set_f_params(A,B,u)
	while integrator.successful() and integrator.t<totalTime:
		sol = np.append(sol,[integrator.integrate(integrator.t+dt)],axis=0)
	return sol[1:,:]

#simulation using euler integration
def network_sim_euler(x0,A,B,C,u,dt,totalTime):
	sol = np.array([x0])
	x = x0
	integrator_t = 0
	while integrator_t<totalTime:
		x += ode_step(integrator_t,x,A,B,u)*dt
		integrator_t += dt
		sol = np.append(sol,[x],axis=0)
	return sol[1:,:]

#to plot intrinsic variables, x
def plot_xs(x0,sol,dt,totalTime):
	t = np.arange(0,totalTime+dt,dt)
	labels = ["x"+str(i) for i in range(0,len(x0))]
	for x_arr,label in zip(sol.transpose(),labels):
		plt.plot(t,x_arr,'.-',label = label)
	plt.legend()
	plt.show()

#to plot observed variables, y
def plot_ys(x0,sol,C,dt,totalTime):
	t = np.arange(0,totalTime+dt,dt)
	y_sol = np.matmul(C,sol.transpose()).transpose()
	labels = ["y"+str(i) for i in range(0,len(x0))]
	for x_arr,label in zip(sol.transpose(),labels):
		plt.plot(t,x_arr,'.-',label = label)
	plt.legend()
	plt.show()



x0 = [1,1]
A = [[1,-5],[10,-1]]
B = [[-1,0],[0,-1]]
C = [[1,0.5],[0.2,1]]
u = [1,0]
dt = 0.1
totalTime = 5

sol = network_sim_euler(x0,A,B,C,u,dt,totalTime)
plot_xs(x0,sol,dt,totalTime)
plot_ys(x0,sol,C,dt,totalTime)

sol2 = network_sim_integrator(x0,A,B,C,u,dt,totalTime)
diff = sol-sol2
plot_xs(x0,sol2,dt,totalTime)
plot_xs(x0,diff,dt,totalTime)