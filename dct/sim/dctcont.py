"""
.. _continuous:

Continuous Simulation (:mod:`cont`)
=========================================

Solving setups of the form,

.. math::
	\dot{x} = \mathbf{A}x + \mathbf{B}u \n
	y = \mathbf{C}x + \mathbf{D}u

Relevant Examples
******************
* :ref:`continuous_eg`
* :ref:`network_eg`

Initialisation and Setting Matrices
************************************

.. autosummary::
	:toctree:

	cont.__init__
	cont.setABC
	cont.setA
	cont.setB
	cont.setC
	cont.setx0
	cont.set_plot_points

Getting Values
***************

.. autosummary::
	:toctree:

	cont.get_x
	cont.get_y
	cont.get_x_set
	cont.get_y_set

Plotting and Saving
********************

.. autosummary::
	:toctree:

	cont.set_plot_points
	cont.plot
	cont.plot_state
	cont.plot_output
	cont.plot_impulse
	cont.plot_step
	cont.plot_comp
	cont.save_state
	cont.save_output
	cont.lqr
	cont.inf_lqr

Simulation Properties
**********************

.. autosummary::
	:toctree:

	cont.is_controllable
	cont.is_observable
	cont.is_stable
"""

import numpy as np
from scipy import linalg
from dct.tools import *

class cont:
	"""Class to contain objects and functions for carrying out continuous simulations of the form,
	

	Attributes:
		A (ndarray): Drift matrix
		B (ndarray): Input matrix
		C (ndarray): Output matrix
		plot_points (int): Number of points to plot when plotting, default is 100
		x0 (ndarray): Initial conditions
	"""
	def __init__(self,n=None,no=None,nu=None):
		"""Generate a cont object.
		
		Args:
			n (int, optional): Dimensions of n x n drift matrix, A
			no (int, optional): Dimension of n x no input matrix, B
			nu (int, optional): Dimensions of nu x n output matrix, C
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
			B (ndarray, optional): Input matrix
			C (ndarray, optional): Output matrix
		
		Returns:
			cont: Updated cont object
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
			cont: Updated cont object
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
			cont: Updated cont object
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
			cont: Updated cont object
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
			cont: Updated cont object
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
			cont: Updated cont object
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

	def get_B_dim(self):
		if(self.ready()):
			dim = np.shape(self.B)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[1]
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
			xs (ndarray, optional): Existing array of state vectors
		
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
		"""Tests if the cont object is controllable.
		
		Returns:
			bool: Boolean, true if the cont configuration is controllable
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
		"""Tests if the cont object is observable.
		
		Returns:
			bool: Boolean, true if the cont configuration is observable
			ndarray: Observability grammian from Lyapunov equation
		"""
		if (self.ready()):
			if(self.C is not None):
				q = -np.matmul(self.C.conj().T,self.C)
				y_o = linalg.solve_lyapunov(self.A.conj().T,q)
				y_o = y_o.conj().T
				observable = (linalg.eigvals(y_o)>0).sum() == np.shape(self.A)[0]
				return [observable,y_o]
			else:
				print("Please set C.")

	def is_stable(self):
		"""Tests if the cont object is stable.

		Returns:
			bool: Boolean, true if the cont configuration is observable
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

	def step(self,time):
		"""
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				a_inv = linalg.inv(self.A)
				s = np.matmul(self.C,np.matmul(a_inv,np.matmul(linalg.expm(time*self.A)-np.identity(np.shape(self.A)[0]),self.B)))
				return s
			else:
				print("Please set A,B and C first.")

	def save_state(self,filename,times,plot_points=None,xs=None):	
		"""Save a set of state vectors.
		
		Args:
			filename (str): Name of file or filepath for save file
			times (array): Array of times of state vectors to be saved
			plot_points (int, optional): Number of points to save, defaults to self.plot_points
			xs (ndarray, optional): Existing set of state vectors to save
		
		Returns:
			cont: To allow for chaining
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
			cont: To allow for chaining
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
		"""Plot both states and outputs (if C is given) of a cont object for a given amount of time.
		
		Args:
			times (array): An array for the form [start time, end time]
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
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
			print(np.shape(x))
			y = self.get_y_set(t,x)
			plot_sio(self,t,False,grid,x,y)
			if(filename  is not None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,times,points,x)
				self.save_output(filename_y,times,points,y)

	def plot_state(self,times,plot_points=None,filename=None,grid=False):
		"""Plot states of a cont object for a given amount of time.
		
		Args:
			times (array): An array for the form [start time, end time]
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
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
		"""Plot outputs (if C is given) of a cont object for a given amount of time.
		
		Args:
			times (array): An array for the form [start time, end time]
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
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

	def plot_impulse(self,times,inputs=None, outputs=None,plot_points=None,filename=None,grid=False):
		"""Plots output responses to input impulses grouped by input.

		Args:
			times (array): An array of the form [start time, end time]
			inputs (array, optional): The inputs to plot, defaults to all inputs
			outputs (array, optional): The outputs to plot, defaults to all outputs
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
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
				plot_resp(self,t,inputs,outputs,False,grid,impulse,"Impulse")
				if(filename is not None):
					return
			else:
				print("Please set A, B and C.")

	def plot_step(self,times,inputs=None, outputs=None,plot_points=None,filename=None,grid=False):
		"""Plots output responses to step inputs grouped by input.

		Args:
			times (array): An array of the form [start time, end time]
			inputs (array, optional): The inputs to plot, defaults to all inputs
			outputs (array, optional): The outputs to plot, defaults to all outputs
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
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
				inv_a = linalg.inv(self.A)
				step = np.array([np.matmul(self.C,np.matmul(inv_a,np.matmul(linalg.expm(self.A*t_i)-np.identity(np.shape(self.A)[0]),self.B))) for t_i in t])
				#step[t,n_c,n_b]
				plot_resp(self,t,inputs,outputs,False,grid,step,"Step")
				if(filename is not None):
					return
			else:
				print("Please set A, B and C.")

	def lqr(self,R,Q,Q_f,times=None,grid=False,plot_points=None):
		"""
		"""
		if(self.ready()):
			if(self.B is not None):
				if(R is None):
					R = 0.2*np.eye(np.shape(self.B)[1])+1e-6
				if(Q is None):
					Q = np.eye(np.shape(self.A)[0])

		return

	def inf_lqr(self,R,Q,times=None,grid=False,plot_points=None):
		"""Computes the infinite horizon linear quadratic regulator given weighting matrices, R and Q. Can plot inputs and state.

		Args:
			R (ndarray): Input weighting matrix
			Q (ndarray): State weighting matrix
			times (array, optional): An array of the form [start time, end time], does not plot if not specified
			plot_points (int, optional): The number of points to use when plotting, default is the internal value, defaulted at 100
			grid (bool, optional): Display grid, default is false

		Returns:
			(tuple): tuple containing:
				
				- P (ndarray): Solution to the continuous algebraic Ricatti equation
				- K (ndarray): The input matrix, u = Kx
		"""
		if(self.ready()):
			if(self.B is not None):
				if(R is None):
					R = 0.2*np.eye(np.shape(self.B)[1])+1e-6
				if(Q is None):
					Q = np.eye(np.shape(self.A)[0])
				P = linalg.solve_continuous_are(self.A,self.B,Q,R)
				K = -np.matmul(linalg.inv(R),np.matmul(self.B.T,P))
				if(times is not None):
					if(plot_points is None):
						plot_points = self.plot_points
					start,end = times
					t = np.linspace(start,end,plot_points)
					x = np.array([np.matmul(linalg.expm((self.A+np.matmul(self.B,K))*t_i),self.x0) for t_i in t])
					u = np.matmul(K,x)
					x = x[:,:,0].T
					u = u[:,:,0].T
					plot_sio(self,t,False,grid,x=x,u=u)
				return (P,K)
			else:
				print("Please set A, B and C using setABC.")

	def plot_comp(self,length=0):
		"""

		"""
		vals,vecs = linalg.eig(self.A)
		d_ratios = -np.real(vals)/np.abs(vals)
		pairs = sorted(zip(d_ratios,vecs.T),key = lambda x: x[0])
		if(length ==2 or length ==4):
			pairs = pairs[::2][0:length]
		elif(length==1):
			pairs = np.array([pairs[0]])
		else:
			if (len(pairs)>=4):
				pairs = pairs[::2][0:4]
			elif(len(pairs)>=2):
				pairs = pairs[::2][0:2]
			else:
				pairs = np.array([pairs[0]])
		compass(pairs)
