"""
.. _network:

Network (:mod:`network`)
=========================

Examples
#########

*:ref:`network_eg`

Power Network Simulation
########################
.. autosummary::
	:toctree:

	network.__init__
	network.generate_cont_sim
	network.generate_disc_sim
	network.show_network

"""
import numpy as np
from dct.sim import *
from dct.tools import *



class network:
	"""Class for representing power networks and generating discrete and continuous simulations of power networks.
	
	Attributes:
	    Adj (ndarray): Adjacency Matrix
	    d (ndarray): Damping Matrix
	    dt (float): Time Step for Discretisation
	    k (ndarray): Coupling Matrix
	    m_inv (ndarray): 1/mass Matrix
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
		toReturn = cont()
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
		C = np.transpose(B)
		toReturn.setABC(A,B=B,C=C)
		return toReturn

	def generate_disc_sim(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		toReturn = disc()
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
		C = np.transpose(B)
		toReturn.setABC(A,B=B,C=C)
		return toReturn

	def show_network(self):
		"""Summary
		"""
		show_graph(self.Adj)
		return self