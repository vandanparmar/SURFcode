"""
.. _discrete:

Discrete Simulation (:mod:`disc`)
==================================
Solving setups of the form,

.. math::
	x[k+1] = \mathbf{A}x[k] + \mathbf{B}u[k] \n
	y[k] = \mathbf{C}x[k] + \mathbf{D}u[k]


Relevant Examples
*****************
* :ref:`discrete_eg`
* :ref:`network_eg`

Initialisation and Setting Matrices
************************************

.. autosummary::
	:toctree:

	disc.__init__
	disc.setABC
	disc.setA
	disc.setB
	disc.setC
	disc.setx0

Getting Values
***************

.. autosummary::
	:toctree:

	disc.get_x
	disc.get_y
	disc.get_x_set
	disc.get_y_set	

Plotting and Saving
********************

.. autosummary::
	:toctree:

	disc.plot
	disc.plot_state
	disc.plot_output
	disc.plot_impulse
	disc.plot_step
	disc.plot_comp
	disc.save_state
	disc.save_output
	disc.lqr
	disc.inf_lqr

Simulation Properties
**********************

.. autosummary::
	:toctree:

	disc.is_controllable
	disc.is_observable
	disc.is_stable
"""
import numpy as np
from scipy import linalg
from dct.tools import *
import cvxpy
import multiprocessing as mp
from itertools import repeat
import time

class disc:
	"""Class to contain objects and functions for carrying out discrete simulations of the form,
	.. math::
		x[k+1] = \mathbf{A}x[k] + \mathbf{B}u[k] \n
		y[k] = \mathbf{C}x[k] + \mathbf{D}u[k]

	where A is the drift matrix, B the input matrix and C the output matrix.

	
	Attributes:
		A (ndarray): Drift Matrix
		B (ndarray): Input Matrix
		C (ndarray): Output Matrix
		x0 (ndarray): Initial Conditions
	"""
	def __init__(self, n=None, no=None, nu=None):
		"""Generate a disc object.
		
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
		"""Set A, B, C for matrices for a discrete simulation.
		
		Args:
			A (ndarray): Drift matrix
			B (ndarray, optional): Input matrix
			C (ndarray, optional): Output matrix
		
		Returns:
			disc: Updated disc object
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
			disc: Updated simulate_cont object
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
			disc: Updated simulate_cont object
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
			disc: Updated simulate_cont object
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
			disc: Updated simulate_cont object
		"""
		if(np.shape(x0)==(np.shape(self.A)[0],1)):
			self.x0 = x0
		else:
			print('x0 dimensions should be',(np.shape(self.A)[0],1),', please try again.')
		return self

	def get_x(self,k):
		"""Calculate a state vector at a particular time.
		
		Args:
			k (int): Index at which to return state vector
		
		Returns:
			ndarray: n x 1 state vector at index value k
		"""
		if(self.ready()):
			x = np.matmul(np.linalg.matrix_power(self.A,k),self.x0)
			return x

	def get_y(self,k):
		"""Calculate an output vector at a particular time.
		
		Args:
			k (int): Index at which to return output vector
		
		Returns:
			ndarray: no x 1 output vector at index k
		"""
		if(self.ready()):
			y = np.matmul(self.C,self.get_x(k))
			return y

	def get_x_set(self,ks):
		"""Calculate a set of x values.
		
		Args:
			ks (array): Array of indices at which to return state vectors
		
		Returns:
			ndarray: n x len(ks) set of state vectors
		"""
		if(self.ready()):
			xs = self.get_x(ks[0])
			x0 = xs
			for time in ks[1:]:
				x0 = np.matmul(self.A,x0)
				xs = np.append(xs,x0,axis=1)
		return xs

	def get_y_set(self,ks,xs=None):
		"""Calculate a set of y values.
		
		Args:
			ks (array): Array of times at which to return output vectors
			xs (ndarray, optional): Existing array of state vectors
		
		Returns:
			ndarray: no x len(ks) set of output vectors
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

	def get_B_dim(self):
		if(self.ready()):
			dim = np.shape(self.B)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[1]
			return toReturn

	def get_C_dim(self):
		if(self.ready()):
			dim = np.shape(self.C)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[0]
			return toReturn

	def is_controllable(self):
		"""Tests if the disc object is controllable.
		
		Returns:
			bool: Boolean, true if the disc configuration is controllable
			ndarray: Controllability gramiam from discrete Lyapunov equation
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
		"""Tests if the disc object is observable.
		
		Returns:
			bool: Boolean, true if the disc configuration is observable
			ndarray: Observability gramiam from discrete Lyapunov equation
		"""
		if (self.ready()):
			if (self.C is not None):
				q = np.matmul(self.C.conj().T,self.C)
				y_o = linalg.solve_discrete_lyapunov(self.A,q)
				y_o = y_o.conj().T
				observable = (linalg.eigvals(y_o)>0).sum() == np.shape(self.A)[0]
				return [observable,y_o]
			else:
				print("Please set C.")

	def is_stable(self):
		"""Tests if the disc object is stable.

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

	def impulse(self,k):
		"""
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				h = np.matmul(np.matmul(self.C,np.linalg.matrix_power(self.A,(k-1))),self.B)
				return h
			else:
				print("Please set A, B and C.")	

	def save_state(self,filename,ks,xs=None):
		"""Save a set of state vectors.
		
		Args:
			filename (str): Name of file or filepath for save file
			ks (array): Array of indices of state vectors to be saved
			xs (ndarray, optional): Existing set fo state vectors to save
		
		Returns:
			disc: To allow for chaining
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
		"""Save a set of output vectors
		
		Args:
			filename (str): Name of file or filepath for save file
			ks (array): Array of indices of output vectors to be saved
			ys (ndarray, optional): Existing set of output vectors to save
		
		Returns:
			disc: To allow for chaining
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
		"""Plot both states and outputs (if C is given) of a disc object for a given set of indices.
		
		Args:
			ks (array): An array for the form [start index, end index]
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
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
		"""Plot states of a disc object for a given set of indices.
		
		Args:
			ks (array): An array for the form [start index, end index]
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
		"""
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end+1)
			x = self.get_x_set(k)
			plot_sio(self,k,True,grid,x=x)
			if(filename is not None):
				self.save_state(filename,ks,x)

	def plot_output(self,ks,filename=None, grid=False):
		"""Plot outputs (if C is given) of a disc object for a given set of indices.
		
		Args:
			ks (array): An array for the form [start index, end index]
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		
		"""
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end+1)
			y = self.get_y_set(k)
			plot_sio(self,k,True,grid,y=y)
			if(filename is not None):
				self.save_state(filename,ks,y)

	def plot_impulse(self,ks,inputs=None, outputs=None,filename=None,grid=False):
		"""Plots output responses to input impulses grouped by input.

		Args:
			ks (array): An array of the form [start index, end index]
			inputs (array, optional): The inputs to plot, defaults to all inputs
			outputs (array, optional): The outputs to plot, defaults to all outputs
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				start,end = ks
				k = np.arange(start,end+1)
				if(inputs is None):
					inputs = np.arange(1,np.shape(self.B)[1]+1)
				if(outputs is None):
					outputs = np.arange(1,np.shape(self.C)[0]+1)
				impulse = np.zeros((len(k),np.shape(self.C)[0],np.shape(self.B)[1]))
				impulse[1:] = np.array([np.matmul(self.C,np.matmul(np.linalg.matrix_power(self.A,k_i-1),self.B)) for k_i in k[1:]])
				#impulse[k,n_c,n_b]
				plot_resp(self,k,inputs,outputs,True,grid,impulse,"Impulse")
				if(filename is not None):
					return
			else:
				print("Please set A, B and C.")

	def plot_step(self,ks,inputs=None, outputs=None,filename=None,grid=False):
		"""Plots output responses to step inputs grouped by input.

		Args:
			ks (array): An array of the form [start index, end index]
			inputs (array, optional): The inputs to plot, defaults to all inputs
			outputs (array, optional): The outputs to plot, defaults to all outputs
			filename (str, optional): Filename to save output to, does not save if none provided
			grid (bool, optional): Display grid, default is false
		"""
		if(self.ready()):
			if(self.B is not None and self.C is not None):
				start,end = ks
				k = np.arange(start,end+1)
				if(inputs is None):
					inputs = np.arange(1,np.shape(self.B)[1]+1)
				if(outputs is None):
					outputs = np.arange(1,np.shape(self.C)[0]+1)
				step = np.zeros((len(k),np.shape(self.C)[0],np.shape(self.B)[1]))
				prev = np.matmul(np.linalg.matrix_power(self.A,k[0]-1),self.B)
				step[0] = np.matmul(self.C,prev)
				for i,k_i in enumerate(k):
					prev = np.matmul(self.A,prev)
					step[i] = step[i-1] + np.matmul(self.C,prev)
				#step[k,n_c,n_b]
				plot_resp(self,k,inputs,outputs,False,grid,step,"Step")
				if(filename is not None):
					return
			else:
				print("Please set A, B and C.")

	def lqr(self,R,Q,Q_f,hor,ks=None,grid=False):
		"""Computes the finite horizon linear quadratic regulator given weighting matrices, R, Q and Q_f. Can plot inputs and state.

		Args:
			R (ndarray): Input weighting matrix
			Q (ndarray): State weighting matrix
			Q_f (ndarray): Final state weighting matrix
			hor (int): Horizon index
			ks (array, optional): An array of the form [start index, end index], does not plot if not specified
			grid (bool, optional): Display grid, default is false

		Returns:
			(tuple): tuple containing:
				- P (ndarray): Stack of matrices solving discrete Ricatti differential equation
				- K (ndarray): Stack of input matrices, u[k] = K[k]x[k]
		"""
		if(self.ready()):
			if(self.B is not None):
				if(R is None):
					R = 0.2*np.eye(np.shape(self.B)[1])+1e-6
				if(Q is None):
					Q = np.eye(np.shape(self.A)[0])
				P = np.array([Q_f])
				K = None
				prev = Q_f
				for hor_i in range(hor,-1,-1):
					APB = np.matmul(self.A.T,np.matmul(prev,self.B))
					inv_R = linalg.inv(R + np.matmul(self.B.T,np.matmul(prev,self.B)))
					K_i = np.array([np.matmul(inv_R,APB.T)])
					if(K is None):
						K = K_i
					else:
						K = np.append(K_i,K,axis=0)
					P_i = Q + np.matmul(self.A.T,np.matmul(prev,self.A)) - np.matmul(APB,np.matmul(inv_R,APB.T))
					P = np.append(np.array([P_i]),P,axis=0)
					prev = P_i
				if(ks is not None):
					start,end = ks
					if (start<0 or end > hor):
						print("Please enter values within the horizon.")
						return (P,K)
					else:
						x = np.array([self.x0])
						u = None
						for i in range(0,hor+1):
							if(u is None):
								u = np.array([np.matmul(K[i],x[i])])
							else:
								#print(K[i])
								u = np.append(u,np.array([np.matmul(K[i],x[i])]),axis=0)
							x = np.append(x,np.array([np.matmul(self.A,x[i])+np.matmul(self.B,u[i])]),axis=0)
						k = np.arange(start,end+1)
						plot_sio(self,k,True,grid,x=x[start:end+1,:,0].T,u=u[start:end+1,:,0].T)
		return (P,K)

	def inf_lqr(self,R,Q,ks=None,grid=False):
		"""Computes the infinite horizon linear quadratic regulator given weighting matrices, R and Q. Can plot inputs and state.

		Args:
			R (ndarray): Input weighting matrix
			Q (ndarray): State weighting matrix
			ks (array, optional): An array of the form [start index, end index], does not plot if not specified
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
				P = linalg.solve_discrete_are(self.A,self.B,Q,R)
				K = -np.matmul(linalg.inv(R+np.matmul(self.B.T,np.matmul(P,self.B))),np.matmul(self.B.T,np.matmul(P,self.A)))
				if (ks is not None):
					start,end = ks
					k = np.arange(start,end+1)
					A_p = self.A + np.matmul(self.B,K)
					x0_p = np.matmul(np.linalg.matrix_power(A_p,start),self.x0)
					x = x0_p
					for k_i in k[1:]:
						x0_p = np.matmul(A_p,x0_p)
						x = np.append(x,x0_p,axis=1)
					u = np.matmul(K,x)
					plot_sio(self,k,True,grid,x=x,u=u)
					plot_hmap(self,k,x,"LQR State","k")
		return (P,K)

	def plot_comp(self,length=0):
		vals,vecs = linalg.eig(self.A)
		d_ratios = -np.real(vals)/np.abs(vals)
		pairs = sorted(zip(d_ratios,vecs.T),key = lambda x: x[0])
		if(length ==2 or length ==4):
			pairs = pairs[0:length]
		elif(length==1):
			pairs = np.array([pairs[0]])
		else:
			if (len(pairs)>=4):
				pairs = pairs[0:4]
			elif(len(pairs)>=2):
				pairs = pairs[0:2]
			else:
				pairs = np.array([pairs[0]])
		compass(pairs)


	def h2(self,hor,C1=None,D12=None,ks=None,heatmap=False,grid=False):
		if(self.ready()):
			if(self.B is not None):
				if(C1 is None):
					C1 = np.eye(np.shape(self.A)[0])
				if(D12 is None):
					D12 = np.eye(np.shape(self.B)[1])
				[n,nu] = np.shape(self.B)
				R = np.array([cvxpy.Variable(n, n) for t in range(0,hor)])
				M = np.array([cvxpy.Variable(nu,n) for t in range(0,hor)])
				def H2_norm(R,M,C,D,T):
					toReturn = 0
					for t in range(0,T):
						toReturn += cvxpy.norm(C*R[t],"fro")
						toReturn += cvxpy.norm(D*M[t],"fro")
					return toReturn
				cost = H2_norm(R,M,C1,D12,hor)
				constr = [R[0]==np.eye(n)]
				for t in range(0,hor-1):
					constr += [R[t+1] == self.A*R[t]+self.B*M[t]]
				constr += [R[hor-1]==0]
				prob = cvxpy.Problem(cvxpy.Minimize(cost),constr)
				t0 = time.time()
				a = prob.solve()
				t1 = time.time()
				print('Minimized value: '+str(a))
				R = np.array(list(map(lambda x: x.value,R)))
				M = np.array(list(map(lambda x: x.value,M))) 
				if(ks is not None):
					times = np.arange(0,ks[1]+1)
					x,u = self.get_x_u_sls(R,M,hor,times)
					if(heatmap):
						plot_hmap(self,times,x,"State","k")
						plot_hmap(self,times,u,"Input","k")
					else:
						times = np.arange(0,ks[1]+1)
						plot_sio(self,times,True,grid,x=x,u=u)
				return [R,M,t1-t0]


	def get_x_u_sls(self,R,M,T,ks,w=None):
		[n,nu] = np.shape(self.B)
		if (w is None):
			w = np.zeros((n,))
			# w = np.matmul(sim.B,np.random.normal(0,0.1,nu))
		x = self.x0
		delta_x = np.zeros((n,T))
		xhat = np.zeros((n,1))
		u = np.zeros((nu,1))
		for k_i in ks:
			delta_x = np.append(delta_x,np.array([x[:,k_i]-xhat[:,k_i]]).T,axis=1)
			for i in range(0,T):
				u[:,k_i]+= np.matmul(M[i],delta_x[:,k_i-i+T])
				xhat[:,k_i] += np.matmul(R[i],delta_x[:,k_i-i+T])
			x_plus1 = np.array([np.matmul(self.A,x[:,-1])+np.matmul(self.B,u[:,-1])+w]).T
			x = np.append(x,x_plus1,axis=1)
			u = np.append(u,np.zeros((nu,1)),axis=1)
			xhat = np.append(xhat,np.zeros((n,1)),axis=1)
		return [x,u]




	def sls_slow(self,hor,d,C1=None,D12=None,ks=None,heatmap=False,grid=False):
		if(self.ready()):
			if(self.B is not None):
				if(C1 is None):
					C1 = np.eye(np.shape(self.A)[0])
				if(D12 is None):
					D12 = np.eye(np.shape(self.B)[1])
				[n,nu] = np.shape(self.B)
				R_struc = [np.linalg.matrix_power(binify(self.A),d-1).tolist() for t in range(0,hor)]
				M_struc = [binify(np.matmul(self.B.T,np.array(R_struc[t]))).tolist() for t in range(0,hor)]
				R = list(map(lambda i: cvxpy.bmat(list(map(lambda j: list(map(lambda k: cvxpy.Variable() if k else 0, j)), i))),R_struc))
				M = list(map(lambda i: cvxpy.bmat(list(map(lambda j: list(map(lambda k: cvxpy.Variable() if k else 0, j)), i))),M_struc))
				def H2_norm(R,M,C,D,T):
					toReturn = 0
					for t in range(0,T):
						toReturn += cvxpy.norm(C*R[t],"fro")
						toReturn += cvxpy.norm(D*M[t],"fro")
					return toReturn
				cost = H2_norm(R,M,C1,D12,hor)
				#print(cost)
				constr = [R[0]==np.eye(n)]
				for t in range(0,hor-1):
					constr += [R[t+1] == self.A*R[t]+self.B*M[t]]
				constr += [R[hor-1]==0]
				prob = cvxpy.Problem(cvxpy.Minimize(cost),constr)
				t0 = time.time()
				a = prob.solve(solver=cvxpy.SCS,eps=1e-10)
				t1 = time.time()
				print('Minimized value: '+str(a))
				R = np.array(list(map(lambda x: x.value,R)))
				M = np.array(list(map(lambda x: x.value,M))) 
				if(ks is not None):
					times = np.arange(0,ks[1])
					x,u = self.get_x_u_sls(R,M,hor,times)
					if(heatmap):
						plot_hmap(self,times,x,"State","k")
						plot_hmap(self,times,u,"Input","k")
					else:
						times = np.arange(0,ks[1]+1)
						plot_sio(self,times,True,grid,x=x,u=u)
				return [R,M]


	def sls(self,hor,d,C1=None,D12=None,ks=None,heatmap=False,grid=False):
		if(self.ready()):
			if(self.B is not None):
				if(C1 is None):
					C1 = np.eye(np.shape(self.A)[0])
				if(D12 is None):
					D12 = np.eye(np.shape(self.B)[1])
				[n,nu] = np.shape(self.B)
				R_struc = [np.linalg.matrix_power(binify(self.A),d-1) for t in range(0,hor)]
				M_struc = [binify(np.matmul(self.B.T,np.array(R_struc[t]))) for t in range(0,hor)]
				R_struc = np.swapaxes(np.array(R_struc),0,2)
				M_struc = np.swapaxes(np.array(M_struc),0,2)
				R_struc = np.swapaxes(R_struc,1,2).tolist()
				M_struc = np.swapaxes(M_struc,1,2).tolist()
				R = list(map(lambda i: (list(map(lambda j: cvxpy.vstack(list(map(lambda k: cvxpy.Variable() if k else 0, j))), i))),R_struc))
				M = list(map(lambda i: (list(map(lambda j: cvxpy.vstack(list(map(lambda k: cvxpy.Variable() if k else 0, j))), i))),M_struc))
				pool = mp.Pool(processes=mp.cpu_count())
				args_list = zip(repeat(self.A),repeat(self.B),R,M,repeat(C1),repeat(D12),repeat(hor),np.eye(n))
				t0 = time.time()
				res = pool.starmap(eval_i,args_list)
				t1 = time.time()
				[R,M] = list(zip(*res))
				R = np.swapaxes(np.array(R)[:,:,:,0],1,2)
				M = np.swapaxes(np.array(M)[:,:,:,0],1,2)
				R = np.swapaxes(R,0,2)
				M = np.swapaxes(M,0,2)
				if(ks is not None):
					times = np.arange(0,ks[1])
					x,u = self.get_x_u_sls(R,M,hor,times)
					if(heatmap):
						plot_hmap(self,times,x,"State","k")
						plot_hmap(self,times,u,"Input","k")
					else:
						times = np.arange(0,ks[1]+1)
						plot_sio(self,times,True,grid,x=x,u=u)
				return [R,M]



	def sls_fast(self,hor,d,C1=None,D12=None,ks=None,heatmap=False,grid=False):
		if(self.ready()):
			if(self.B is not None):
				if(C1 is None):
					C1 = np.eye(np.shape(self.A)[0])
				if(D12 is None):
					D12 = np.eye(np.shape(self.B)[1])
				[n,nu] = np.shape(self.B)
				A = self.A
				B = self.B
				R_struc = np.linalg.matrix_power(binify(self.A),d-1)
				M_struc = binify(np.matmul(self.B.T,np.array(R_struc)))
				cols_r = []
				cols_m = []
				As = []
				Bs = []
				Cs = []
				Ds = []
				eye = []
				for col in range(0,n):
					cols_r.append(np.nonzero(R_struc[:,col])[0])
					col_r_n = np.shape(cols_r[col][0])
					cols_m.append(np.nonzero(M_struc[:,col])[0])
					col_m_n = np.shape(cols_m[col][0])
					s_r = set()
					for row in cols_r[col]:
						s_r.update(np.nonzero(A[:,row])[0].tolist())	
						s_r.update(np.nonzero(C1[:,row])[0].tolist())	
					for row in cols_m[col]:
						s_r.update(np.nonzero(B[:,row])[0].tolist())
						s_r.update(np.nonzero(D12[:,row])[0].tolist())
					this_A = np.zeros((len(s_r),len(s_r)))
					this_B = np.zeros((len(s_r),len(cols_m[col])))
					this_C = np.zeros((len(s_r),len(s_r)))
					this_D = np.zeros((len(s_r),len(cols_m[col])))
					for i,row in enumerate(cols_r[col]):
						this_A[:,i] = A[sorted(s_r),row]
						this_C[:,i] = C1[sorted(s_r),row]
					for i,row in enumerate(cols_m[col]):
						this_B[:,i] = B[sorted(s_r),row]
						this_D[:,i] = D12[sorted(s_r),row]
					this_eye = np.zeros((len(s_r),))
					this_eye[cols_r[col].tolist().index(col)]=1
					eye.append(this_eye)
					As.append(this_A)
					Bs.append(this_B)
					Cs.append(this_C)
					Ds.append(this_D)
				pool = mp.Pool(processes=mp.cpu_count())
				args_list = zip(As,Bs,Cs,Ds,cols_r,cols_m,repeat(hor),eye,repeat(n),repeat(nu))
				t0 = time.time()
				res = pool.starmap(eval_i_fast,args_list)
				t1 = time.time()
				[R,M] = list(zip(*res))
				R = np.swapaxes(np.array(R)[:,:,:],1,2)
				M = np.swapaxes(np.array(M)[:,:,:],1,2)
				R = np.swapaxes(R,0,2)
				M = np.swapaxes(M,0,2)
				if(ks is not None):
					times = np.arange(0,ks[1])
					x,u = self.get_x_u_sls(R,M,hor,times)
					if(heatmap):
						plot_hmap(self,times,x,"State","k")
						plot_hmap(self,times,u,"Input","k")
					else:
						times = np.arange(0,ks[1]+1)
						plot_sio(self,times,True,grid,x=x,u=u)
				return [R,M]

def eval_i_fast(A,B,C,D,col_r,col_m,T,eye,n,nu):
	R = [cvxpy.Variable(np.shape(A)[0],1) for t in range(0,T)]
	M = [cvxpy.Variable(np.shape(B)[1],1) for t in range(0,T)]
	eye_R = np.zeros(np.shape(A)[0])
	eye_R[len(col_r):]=1
	cost = H2_fast(R,M,C,D,T)
	constr = [R[0]==eye]
	for t in range(0,T-1):
		constr += [R[t+1]==A*R[t]+B*M[t]]
		constr += [cvxpy.mul_elemwise(eye_R,R[t])==0]
	constr += [R[T-1]==0]
	prob = cvxpy.Problem(cvxpy.Minimize(cost),constr)
	a = prob.solve()
	print(a)
	R_min = np.array(list(map(lambda x: x.value,R)))[:,:,0]
	M_min = np.array(list(map(lambda x: x.value,M)))[:,:,0]
	R = np.zeros((T,n))
	R[:,col_r] = R_min[:,0:len(col_r)]
	M = np.zeros((T,nu))
	M[:,col_m] = M_min[:,:]
	return (R,M)


def H2_fast(R,M,C,D,T):
	toReturn = 0
	for t in range(0,T):
		toReturn += cvxpy.norm(C*R[t],"fro")
		toReturn += cvxpy.norm(D*M[t],"fro")
	return toReturn


def eval_i(A,B,R,M,C1,D12,T,eye):
	cost = H2(R,M,C1,D12,T)
	constr = [R[0]==eye]
	for t in range(0,T-1):
		constr += [R[t+1]==A*R[t]+B*M[t]]
	constr += [R[T-1]==0]
	prob = cvxpy.Problem(cvxpy.Minimize(cost),constr)
	a = prob.solve()
	R = np.array(list(map(lambda x: x.value,R)))
	M = np.array(list(map(lambda x: x.value,M)))
	return (R,M)


def H2(R,M,C,D,T):
	toReturn = 0
	for t in range(0,T):
		vec1 = C*R[t]
		vec2 = D*M[t]
		toReturn += cvxpy.norm(vec1)
		toReturn += cvxpy.norm(vec2)
	return toReturn

def binify(A):
	toReturn = np.array((A!=0.0))
	return toReturn
