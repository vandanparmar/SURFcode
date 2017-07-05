import numpy as np

class Generator:
	
	def __init__():
		self.th = 0
		self.om = 0
		self.k = 1
		self.d = 1
		self.m = 1
		self.dt = 0.2
		self.A = np.array([[1,self.dt],[-self.k/self.m * self.dt,1-self.d/self.m * self.dt]])
		self.B = np.array([0,1])
		self.C = np.array([])
		self.neighbours = np.array([])
		return

	def __init__(self,theta,omega):
		self.th = theta
		self.om = omega
		self.k = 1
		self.d = 1
		self.m = 1
		self.dt = 0.2
		self.A = np.array([[1,self.dt],[-self.k/self.m * self.dt,1-self.d/self.m * self.dt]])
		self.B = np.array([0,1])
		self.C = np.array([])
		self.neighbours = np.array([])
		return

	def set_k(self,k):
		self.k = k
		return

	def set_m(self,m):
		self.m = m
		return

	def set_d(self,d):
		self.d = d
		return


	def add_neighbour(self,n):
		self.neighbours.append(n)
		return