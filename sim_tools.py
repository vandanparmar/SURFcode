"""
Simulation Tools (:mod:`sim_tools`)
==============================================

This module contains simulations tools for both :ref:`discrete` and :ref:`continuous`. 


Matrix Generation
###################

Functions to generate types of matrix for simulations.

.. autosummary::
	:toctree:

	random_mat
	random_stable
	random_unit


.. _continuous:

Continuous Simulations
########################


Solving setups of the form,

.. math::
	\dot{x} = \mathbf{A}x + \mathbf{B}u \n
	y = \mathbf{C}x + \mathbf{D}u



Initialisation and Setting Matrices
************************************

.. autosummary::
	:toctree:

	simulate_cont.__init__
	simulate_cont.setABC
	simulate_cont.setA
	simulate_cont.setB
	simulate_cont.setC
	simulate_cont.setx0
	simulate_cont.set_plot_points

Getting Values
***************

.. autosummary::
	:toctree:

	simulate_cont.get_x
	simulate_cont.get_y
	simulate_cont.get_x_set
	simulate_cont.get_y_set

Plotting and Saving
********************

.. autosummary::
	:toctree:

	simulate_cont.set_plot_points
	simulate_cont.plot
	simulate_cont.plot_state
	simulate_cont.plot_output
	simulate_cont.save_state
	simulate_cont.save_output

Simulation Properties
**********************

.. autosummary::
	:toctree:

	simulate_cont.is_controllable
	simulate_cont.is_observable
	simulate_cont.is_stable


.. _discrete:

Discrete Simulations
########################


Solving setups of the form,

.. math::
	x[k+1] = \mathbf{A}x[k] + \mathbf{B}u[k] \n
	y[k] = \mathbf{C}x[k] + \mathbf{D}u[k]



Initialisation and Setting Matrices
************************************

.. autosummary::
	:toctree:

	simulate_disc.__init__
	simulate_disc.setABC
	simulate_disc.setA
	simulate_disc.setB
	simulate_disc.setC
	simulate_disc.setx0

Getting Values
***************

.. autosummary::
	:toctree:

	simulate_disc.get_x
	simulate_disc.get_y
	simulate_disc.get_x_set
	simulate_disc.get_y_set

Plotting and Saving
********************

.. autosummary::
	:toctree:

	simulate_disc.plot
	simulate_disc.plot_state
	simulate_disc.plot_output
	simulate_disc.save_state
	simulate_disc.save_output

Simulation Properties
**********************

.. autosummary::
	:toctree:

	simulate_disc.is_controllable
	simulate_disc.is_observable
	simulate_disc.is_stable
"""


#__all__ = ["random_mat","random_stable","random_unit"]

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from graph_tools import *



def random_mat(a,b):
	"""Generate a random a x b matrix.

	Args:
	    a (int): Dimension 1
	    b (int): Dimension 2

	Returns:
	    ndarray: a x b matrix with random entries uniformly distributed on [0,1)
	"""
	toReturn = np.random.rand(a,b)
	return toReturn

def random_stable(n):
	"""Generate a random n x n stable matrix.
	
	Args:
	    n (int): Dimension of matrix
	
	Returns:
	    ndarray: n x n matrix with random entries uniformly distributed on [0,1), eigenvalues all have negative real part
	"""
	toReturn = np.random.rand(n,n)*2-1
	while (np.real(linalg.eigvals(toReturn))<=0).sum() != n:
		toReturn = np.random.rand(n,n)*2-1
	return toReturn

def random_unit(n):
	"""Generate a random n x n matrix with spectral radius less than 1.
	
	Args:
	    n (int): Dimensions of matrix
	
	Returns:
	    ndarray: n x n matrix with random entries uniformly distributed on [0,1), eigenvalues lie in the unit circle
	"""
	toReturn = np.random.rand(n,n)*2-1
	if((np.abs(linalg.eigvals(toReturn))<=1).sum() != n):
		max_e = np.max(np.abs(linalg.eigvals(toReturn)))
		toReturn /= max_e*np.random.uniform(1,5)
	return toReturn

def plot_sio(self,times,disc,grid, x=None, y=None,u=None):
	if(disc):
		xlabel = "K"
		title = " plot for k = "+str(times[0])+' to k = '+str(times[-1])+'.'
	else:
		xlabel = "Time"
		title = " plot for t = "+str(times[0])+' to t = '+str(times[-1])+'.'
	plot_type = []
	if(x is not None):
		plot_type.append(('x','State',len(self.x0)))
	if(y is not None):
		plot_type.append(('y','Output',self.get_C_dim()))
	if(u is not None):
		plot_type.append(('u','Input',self.get_B_dim()))
	plot_tot = len(plot_type)
	for index,var in enumerate(plot_type):
		labels = [var[0]+str(i) for i in range(1,var[2]+1)]
		plt.subplot(plot_tot,1,index+1)
		for arr,label in zip(eval(var[0]),labels):
			if(disc):
				plt.step(times,arr,label=label)
			else:
				plt.plot(times,arr,label=label)
		plt.title(var[1]+title)
		plt.ylabel(var[1])
		plt.xlabel(xlabel)
		plt.legend()
		if(grid):
			plt.grid(color = '#a6a5a6')
	plt.subplots_adjust(hspace = 0.5)
	plt.show()

def plot_resp(self,times,inputs,outputs,disc,grid,resp):
	if(disc):
		xlabel = "K"
		title = " for k = "+str(times[0])+' to k = '+str(times[-1])+'.'
	else:
		xlabel = "Time"
		title = ' for t = '+str(times[0])+' to t = '+str(times[-1])+'.'
	t,n_c,n_b = np.shape(resp)
	for i,input_i in enumerate(inputs):
		plt.subplot(len(inputs),1,i+1)
		for o,output_i in enumerate(outputs):
			label = "Output "+str(output_i)
			if(disc):
				plt.step(times,resp[:,output_i-1,input_i-1],label=label)
			else:
				plt.plot(times,resp[:,output_i-1,input_i-1],label=label)
		plt.title("Impulse response from input "+str(input_i)+title)
		plt.ylabel("Output")
		plt.xlabel(xlabel)
		plt.legend()
		if(grid):
			plt.grid(color= '#a6a5a6')
	plt.subplots_adjust(hspace=0.5)
	plt.show()

class simulate_cont:
	"""Class to contain objects and functions for carrying out continuous simulations of the form,
	

	Attributes:
	    A (ndarray): Drift matrix
	    B (ndarray): Input matrix
	    C (ndarray): Output matrix
	    plot_points (int): Number of points to plot when plotting, default is 100
	    x0 (ndarray): Initial conditions
	"""
	def __init__(self,n=None,no=None,nu=None):
		"""Generate a simulate_cont object.
		
		Args:
		    n (None, optional): Dimensions of n x n drift matrix, A
		    no (None, optional): Dimension of n x no input matrix, B
		    nu (None, optional): Dimensions of nu x n output matrix, C
		"""
		if(n is None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = None
			self.__ready = False
		else:
			self.__ready = True
			self.A = random_stable(n)
			if(no is None):
				self.C = None
			else:
				self.C = random_mat(no,n)
			if(nu is None):
				self.B = None
			else:
				self.B = random_mat(n,nu)
			self.x0 = np.random.rand(n,1)
		self.plot_points = 100

	def setABC(self,A,B=None,C=None):
		"""Set A, B, C matrices for a continuous simulation.
		
		Args:
		    A (ndarray): Drift matrix
		    B (None, optional): Input matrix
		    C (None, optional): Output matrix
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		shapeA = np.shape(A)
		if(shapeA[0] == shapeA[1]):
			self.A = np.array(A)
			n = shapeA[0]
			self.x0 = np.random.rand(n,1)
			self.__ready = True

		else:
			print('Please supply a square A matrix.')

		if(C is not None):
			if(np.shape(C)[1]==n):
				self.C = C
			elif(np.shape(C)[0]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')

		if(B  is not None):
			if(np.shape(B)[0]==n):
				self.B = np.array(B)
			elif(np.shape(B)[1]==n):
			#	self.B = np.transpose(np.array(B))
				print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
		return self

	def ready(self):
		if(self.__ready):
			return True
		else:
			print('Please set A, B and C using setABC.')
			return False

	def setA(self,A):
		"""Set drift matrix, A.
		
		Args:
		    A (ndarray): n x n drift matrix, A
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		if(self.C  is not None):
			if(np.shape(A)[0]==np.shape(self.C)[0]):
				self.A = np.array(A)
			else:
				print('Dimensions of A not compatible, please try again.')
		else:
			print('Please set A, B and C using setABC.')
		return self

	def setB(self,B):
		"""Set input matrix, B.
		
		Args:
		    B (ndarray): n x no input matrix, B
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		n = np.shape(self.A)[0]
		if(np.shape(B)[0]==n):
			self.B = np.array(B)
		elif(np.shape(B)[1]==n):
		#	self.B = np.transpose(np.array(B))
			print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
		return self

	def setC(self,C):
		"""Set output matrix, C.
		
		Args:
		    C (ndarray): nu x n output matrix, C
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		n = np.shape(self.A)[0]
		if(np.shape(C)[1]==n):
			self.C = np.array(C)
		elif(np.shape(C)[0]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		return self

	def setx0(self,x0):
		"""Set intital conditions, x0.
		
		Args:
		    x0 (ndarray): n x 1 initial conditions, x0
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		if(np.shape(x0)==(np.shape(self.A)[0],1)):
			self.x0 = x0
		else:
			print('x0 dimensions should be',(np.shape(self.A)[0],1),', please try again.')
		return self

	def set_plot_points(self,points):
		"""Set number of points to use when plotting, plot_points.
		
		Args:
		    points (int): The number of points to use
		
		Returns:
		    simulate_cont: Updated simulate_cont object
		"""
		if(points<10000):
			self.plot_points = points
		return self

	def get_x(self,t):
		"""Calculate a state vector at a particular time.
		
		Args:
		    t (int): Time at which to return state vector
		
		Returns:
		    ndarray: n x 1 state vector at time t
		"""
		if(self.ready()):
			x = np.matmul(linalg.expm(self.A*t),self.x0)
			return x

	def get_y(self,t):
		"""Calculate an output vector at a particular time.
		
		Args:
		    t (int): Time at which to return output vector
		
		Returns:
		    ndarray: no x 1 output vector at time t 
		"""
		if(self.ready()):
			y = np.matmul(self.C,self.get_x(t))
			return y

	def get_C_dim(self):
		if(self.ready()):
			dim = np.shape(self.C)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[0]
			return toReturn

	def get_x_set(self,times):
		"""Calculate a set of x values.
		
		Args:
		    times (array): Array of times at which to return state vectors
		
		Returns:
		    ndarray: n x len(times) set of state vectors
		"""
		if(self.ready()):
			xs = self.get_x(times[0])
			for time in times[1:]:
				xs = np.append(xs,self.get_x(time),axis=1)
			return xs

	def get_y_set(self,times,xs=None):
		"""Calculate a set of y values.
		
		Args:
		    times (array): Array of times at which to return output vectors
		    xs (None, optional): Existing array of state vectors
		
		Returns:
		    ndarray: n0 x len(times) set of output vectors
		"""
		if(self.ready()):
			if(xs is None):
				ys = self.get_y(times[0])
				for time in times[1:]:
					ys = np.append(ys,self.get_y(time),axis=1)
			else:
				ys = np.matmul(self.C,xs)		
		return ys

	def is_controllable(self):
		"""Tests if the simulate_cont object is controllable.
		
		Returns:
		    bool: Boolean, true if the simulate_cont configuration is controllable
		    ndarray: Controllability grammian from Lyapunov equation
		"""
		if (self.ready()):
			if (self.B is not None):
				q = -np.matmul(self.B,self.B.conj().T)
				x_c = linalg.solve_lyapunov(self.A,q)
				controllable = (linalg.eigvals(x_c)>0).sum() == np.shape(self.A)[0]
				return [controllable,x_c]
			else:
				print("Please set B.")

	def is_observable(self):
		"""Tests if the simulate_cont object is observable.
		
		Returns:
			bool: Boolean, true if the simulate_cont configuration is observable
			ndarray: Observability grammian from Lyapunov equation
		"""
		if (self.ready()):
			if(self.C is not None):
				q = -np.matmul(self.C,self.C.conj().T)
				y_o = linalg.solve_lyapunov(self.A.conj().T,q)
				y_o = y_o.conj().T
				controllable = (linalg.eigvals(y_o)>0).sum() == np.shape(self.A)[0]
				return [controllable,y_o]
			else:
				print("Please set C.")

	def is_stable(self):
		"""Tests if the simulate_cont object is stable.

		Returns:
			bool: Boolean, true if the simulate_cont configuration is observable
			array: The eigenvalues of the A matrix
		"""
		if (self.ready()):
			eigs = linalg.eigvals(self.A)
			toReturn = False
			if ((np.real(eigs)<=0).sum()) == np.shape(self.A)[0]:
				toReturn = True
			return [toReturn,eigs]

	def impulse(self,time):
		"""
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				h = np.matmul(np.matmul(self.C,np.expm(self.A*time)),self.B)
				return h
			else:
				print("Please set A, B and C.")	

	def plot_impulse(self,times,inputs=None, outputs=None,plot_points=None,filename=None,grid=False):
		"""Group by inputs, select arrays of inputs / outputs.
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				start,end = times
				t = np.linspace(start,end,self.plot_points)
				if(inputs is None):
					inputs = np.arange(1,np.shape(self.B)[1]+1)
				if(outputs is None):
					outputs = np.arange(1,np.shape(self.C)[0]+1)
				if(plot_points is None):
					plot_points = self.plot_points
				impulse = np.array([np.matmul(self.C,np.matmul(linalg.expm(self.A*t_i),self.B)) for t_i in t])
				#impulse[t,n_c,n_b]
				plot_resp(self,t,inputs,outputs,False,grid,impulse)
				if(filename is not None):
					return
			else:
				print("Please set A, B and C.")

	def save_state(self,filename,times,plot_points=None,xs=None):	
		"""Save a set of state vectors.
		
		Args:
		    filename (str): Name of file or filepath for save file
		    times (array): Array of times of state vectors to be saved
		    plot_points (int, optional): Number of points to save, defaults to self.plot_points
		    xs (ndarray, optional): Existing set of state vectors to save
		
		Returns:
		    simulate_cont: To allow for chaining
		"""
		if(self.ready()):
			if(plot_points is None):
				plot_points = self.plot_points
			eigvals = linalg.eigvals(self.A)
			start,end = times
			if(xs is None):
				self.get_x_set(times)
			if(len(xs)>10000):
				print('Too many states to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,xs,header=comment)	
		return self

	def save_output(self,filename,times,plot_points=None,ys=None):
		"""Save a set of output vectors
		
		Args:
		    filename (str): Name of file or filebath for save file
		    times (int): Array of times of output vectors to be saved
		    plot_points (int, optional): Number of points to save, defaults to self.plot_points
		    ys (ndarray, optional): Existing set of output vectors to save
		
		Returns:
		    simulate_cont: To allow for chaining
		"""
		if(self.ready()):
			if(plot_points is None):
				plot_points = self.plot_points
			eigvals = linalg.eigvals(self.A)
			start,end = times
			if(ys is None):
				self.get_y_set(times)
			if(len(ys)>10000):
				print('Too many outputs to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,ys,header=comment)			
		return self

	def plot(self,times,plot_points=None,filename=None,grid=False):
		"""Summary
		
		Args:
		    times (TYPE): Description
		    plot_points (None, optional): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			if(self.C is None):
				self.plot_state(times,plot_points,filename,grid)
				return
			if(plot_points is None):
				plot_points = self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			x = self.get_x_set(t)
			y = self.get_y_set(t,x)
			plot_sio(self,t,False,grid,x,y)
			if(filename  is not None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,times,points,x)
				self.save_output(filename_y,times,points,y)

	def plot_state(self,times,plot_points=None,filename=None,grid=False):
		"""Summary
		
		Args:
		    times (TYPE): Description
		    plot_points (None, optional): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		"""
		if(self.ready()):
			if(plot_points is None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			x = self.get_x_set(t)
			plot_sio(self,t,False,grid,x=x)
			if(filename  is not None):
				self.save_state(filename,times,points,x)

	def plot_output(self,times,plot_points=None,filename=None,grid=False):
		"""Summary
		
		Args:
		    times (TYPE): Description
		    plot_points (None, optional): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		"""
		if(self.ready()):
			if(plot_points is None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			y = self.get_y_set(t)
			plot_sio(self,t,False,grid,y=y)
			if(filename  is not None):
				self.save_output(filename,times,points,y)

class simulate_disc:
	"""Class to contain objects and functions for carrying out discrete simulations of the form,
	.. math::
		x[k+1] = \mathbf{A}x[k] + \mathbf{B}u[k] \n
		y[k] = \mathbf{C}x[k] + \mathbf{D}u[k]

	where A is the drift matrix, B the input matrix and C the output matrix.

	
	Attributes:
	    A (TYPE): Drift Matrix
	    B (TYPE): Input Matrix
	    C (TYPE): Output Matrix
	    x0 (TYPE): Initial Conditions
	"""
	def __init__(self, n=None, no=None, nu=None):
		"""Generate a simulate_disc object.
		
		Args:
		    n (None, optional): Dimensions of n x n drift matrix, A
		    no (None, optional): Dimension of n x no input matrix, B
		    nu (None, optional): Dimensions of nu x n output matrix, C
		"""
		if(n is None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = None
			self.__ready = False
		else:
			self.__ready = True
			self.A = random_unit(n)
			if(no is None):
				self.C = None
			else:
				self.C = random_mat(no,n)
			if(nu is None):
				self.B = None
			else:
				self.B = random_mat(n,nu)
			self.x0 = np.random.rand(n,1)			

	def setABC(self,A,B=None,C=None):
		"""Summary
		
		Args:
		    A (TYPE): Description
		    B (None, optional): Description
		    C (None, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		shapeA = np.shape(A)
		if(shapeA[0] == shapeA[1]):
			self.A = np.array(A)
			n = shapeA[0]
			self.x0 = np.random.rand(n,1)
			self.__ready = True
		else:
			print('Please supply a square A matrix.')
		
		if(C  is not None):
			if(np.shape(C)[1]==n):
				self.C = np.array(C)
			elif(np.shape(C)[0]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		
		if(B  is not None):
			if(np.shape(B)[0]==n):
				self.B = np.array(B)
			elif(np.shape(B)[1]==n):
			#	self.B = np.transpose(np.array(B))
				print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
		return self

	def ready(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		if(self.__ready):
			return True
		else:
			print('Please set A, B and C using setABC.')
			return False

	def setA(self,A):
		"""Summary
		
		Args:
		    A (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.C  is not None):
			if(np.shape(A)[0]==np.shape(self.C)[0]):
				self.A = np.array(A)
			else:
				print('Dimensions of A not compatible, please try again.')
		else:
			print('Please set A, B and C using setABC.')
		return self

	def setB(self,B):
		"""Summary
		
		Args:
		    B (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		n = np.shape(self.A)[0]
		if(np.shape(B)[0]==n):
			self.B = np.array(B)
		elif(np.shape(B)[1]==n):
		#	self.B = np.transpose(np.array(B))
			print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
		return self

	def setC(self,C):
		"""Summary
		
		Args:
		    C (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		n = np.shape(self.A)[0]
		if(np.shape(C)[1]==n):
			self.C = np.array(C)
		elif(np.shape(C)[0]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		return self

	def setx0(self,x0):
		"""Summary
		
		Args:
		    x0 (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		if(np.shape(x0)==(np.shape(self.A)[0],1)):
			self.x0 = x0
		else:
			print('x0 dimensions should be',(np.shape(self.A)[0],1),', please try again.')
		return self

	def get_x(self,k):
		"""Summary
		
		Args:
		    k (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			x = np.matmul(np.linalg.matrix_power(self.A,k),self.x0)
			return x

	def get_y(self,k):
		"""Summary
		
		Args:
		    k (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			y = np.matmul(self.C,self.get_x(k))
			return y

	def get_x_set(self,ks):
		"""Summary
		
		Args:
		    ks (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			xs = self.get_x(ks[0])
			x0 = xs
			for time in ks[1:]:
				x0 = np.matmul(self.A,x0)
				xs = np.append(xs,x0,axis=1)
		return xs

	def get_y_set(self,ks,xs=None):
		"""Summary
		
		Args:
		    ks (TYPE): Description
		    xs (None, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			if(xs is None):
				x_0 = self.get_x(ks[0])
				ys = np.matmul(self.C,x_0)
				for time in ks[1:]:
					x_0 = np.matmul(self.A,x_0)
					ys = np.append(ys,np.matmul(self.C,x_0),axis=1)
			else:
				ys = np.matmul(self.C,xs)		
		return ys

	def get_C_dim(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			dim = np.shape(self.C)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[0]
			return toReturn

	def is_controllable(self): #should be reachable?
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		if (self.ready()):
			if (self.B is not None):
				q = np.matmul(self.B,self.B.conj().T)
				x_c = linalg.solve_discrete_lyapunov(self.A.conj().T,q)
				controllable = (linalg.eigvals(x_c)>0).sum() == np.shape(self.A)[0]
				return [controllable,x_c]
			else:
				print("Please set B.")

	def is_observable(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		if (self.ready()):
			if (self.C is not None):
				q = np.matmul(self.C,self.C.conj().T)
				y_o = linalg.solve_discrete_lyapunov(self.A,q)
				observable = (linalg.eigvals(y_o)>0).sum() == np.shape(self.A)[0]
				return [observable,y_o]
			else:
				print("Please set C.")

	def is_stable(self):
		"""Tests if the simulate_disc object is stable.

		Returns:
			bool: Boolean, true if the simulate_cont configuration is observable
			array: The eigenvalues of the A matrix
		"""
		if(self.ready()):
			eigs = linalg.eigvals(self.A)
			toReturn = False
			if ((np.abs(eigs)<1).sum()) == np.shape(self.A)[0]:
				toReturn = True
			return [toReturn,eigs]

	def save_state(self,filename,ks,xs=None):
		"""Summary
		
		Args:
		    filename (TYPE): Description
		    ks (TYPE): Description
		    xs (None, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			eigvals = linalg.eigvals(self.A)
			start,end = ks
			if(xs is None):
				self.get_x_set(ks)
			if(len(xs)>10000):
				print('Too many states to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,xs,header=comment)
		return self

	def save_output(self,filename,ks,ys=None):
		"""Summary
		
		Args:
		    filename (TYPE): Description
		    ks (TYPE): Description
		    ys (None, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			eigvals = linalg.eigvals(self.A)
			start,end = ks
			if(ys is None):
				self.get_y_set(ks)
			if(len(ys)>10000):
				print('Too many outputs to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,ys,header=comment)
		return self

	def plot(self,ks,filename=None, grid=False):
		"""Summary
		
		Args:
		    ks (TYPE): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		
		Returns:
		    TYPE: Description
		"""
		if(self.ready()):
			if(self.C is None):
				self.plot_state(ks,filename,grid)
				return
			start,end = ks
			k = np.arange(start,end+1)
			x = self.get_x_set(k)
			y = self.get_y_set(k,x)
			plot_sio(self,k,True,grid,x=x,y=y)
			if(filename is not None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,ks,x)
				self.save_output(filename_y,ks,y)

	def plot_state(self,ks,filename=None, grid=False):
		"""Summary
		
		Args:
		    ks (TYPE): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		"""
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end+1)
			x = self.get_x_set(k)
			plot_sio(self,k,True,grid,x=x)
			if(filename is not None):
				self.save_state(filename,ks,x)

	def plot_output(self,ks,filename=None, grid=False):
		"""Summary
		
		Args:
		    ks (TYPE): Description
		    filename (None, optional): Description
		    grid (bool, optional): Description
		"""
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end+1)
			y = self.get_y_set(k)
			plot_sio(self,k,True,grid,y=y)
			if(filename is not None):
				self.save_state(filename,ks,y)

class power_network:
	"""Class for representing power networks and generating discrete and continuous simulations of power networks.
	
	Attributes:
	    Adj (TYPE): Adjacency Matrix
	    d (TYPE): Damping Matrix
	    dt (float): Time Step for Discretisation
	    k (TYPE): Coupling Matrix
	    m_inv (TYPE): 1/mass Matrix
	"""
	def __init__(self,Adj):
		"""Summary
		
		Args:
		    Adj (TYPE): Description
		"""
		self.Adj = Adj
		n = np.shape(Adj)[0]
		self.k = np.random.rand(n,n)/2+0.5
		self.m_inv = np.random.rand(n,1)/2+1.0
		self.d = np.random.rand(n,1)*2
		self.dt = 0.2

	def generate_cont_sim(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		toReturn = simulate_cont()
		n = np.shape(self.Adj)[0]
		lap = -generate_laplacian(self.Adj)
		A = np.zeros((2*n,2*n))
		B = np.zeros((2*n,n))
		k_set = np.multiply(self.k,lap) #elementwise multiplication
		k_set = np.multiply(self.m_inv,k_set)*self.dt
		eye_n = np.eye(n)
		d_m_set = -np.matmul(np.diag(self.d.flatten()),np.diag(self.m_inv.flatten()))
		d_m_set += eye_n
		A[::2,::2] = eye_n
		A[1::2,::2] = k_set
		A[::2,1::2] = self.dt*eye_n
		A[1::2,1::2] = d_m_set
		B[1::2,:] = eye_n
		toReturn.setABC(A,B)
		return toReturn

	def generate_disc_sim(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		toReturn = simulate_disc()
		n = np.shape(self.Adj)[0]
		lap = -generate_laplacian(self.Adj)
		A = np.zeros((2*n,2*n))
		B = np.zeros((2*n,n))
		k_set = np.multiply(self.k,lap) #elementwise multiplication
		k_set = np.multiply(self.m_inv,k_set)*self.dt
		eye_n = np.eye(n)
		d_m_set = -np.matmul(np.diag(self.d.flatten()),np.diag(self.m_inv.flatten()))
		d_m_set += eye_n
		A[::2,::2] = eye_n
		A[1::2,::2] = k_set
		A[::2,1::2] = self.dt
		A[1::2,1::2] = d_m_set
		B[1::2,:] = eye_n
		toReturn.setABC(A,B)
		return toReturn
