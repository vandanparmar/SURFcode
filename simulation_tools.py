import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from graph_tools import *

def random_mat(a,b):
	toReturn = np.random.rand(a,b)
	return toReturn

def random_nsd(n):
	toReturn = np.random.rand(n,n)*2-1
	while (np.real(linalg.eigvals(toReturn))<=0).sum() != n:
		toReturn = np.random.rand(n,n)*2-1
	return toReturn

class simulate_cont:
	def __init__(self,n=None,n0=None,nu=None):
		if(n == None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = np.array([])
		else:
			self.A = random_nsd(n)
			if(n0 == None):
				self.C = np.identity(n)
			else:
				self.C = random_mat(n,n0)
			if(nu == None):
				self.B = None
			else:
				self.B = random_mat(n,nu)
			self.x0 = np.random.rand(n)
		self.plot_points = 100

	def setABC(self,A,C=None,B=None):
		shapeA = np.shape(A)
		if(shapeA[0] == shapeA[1]):
			self.A = np.array(A)
			n = shapeA[0]
			self.x0 = np.random.rand(n)
		else:
			print('Please supply a square A matrix.')
			return

		if(C != None):
			if(np.shape(C)[0]==n):
				self.C = np.array(C)
			elif(np.shape(C)[1]==n):
				self.C = np.transpose(np.array(C))
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
				return
				
		if(B != None):
			if(np.shape(B)[0]==n):
				self.B = np.array(B)
			elif(np.shape(B)[1]==n):
				self.B = np.transpose(np.array(B))
			else:
				print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
				return

	def setA(self,A):
		if(np.shape(A)[0]==np.shape(self.C)[0]):
			self.A = np.array(A)
			self.x0 = np.random.rand(np.shape(A)[0])
		else:
			print('Dimensions of A not compatible, please try again.')
		return

	def setB(self,B):
		n = np.shape(self.A)[0]
		if(np.shape(B)[0]==n):
			self.B = np.array(B)
		elif(np.shape(B)[1]==n):
			self.B = np.transpose(np.array(B))
		else:
			print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
			return

	def setC(self,C):
		n = np.shape(self.A)[0]
		if(np.shape(C)[0]==n):
			self.C = np.array(C)
		elif(np.shape(B)[1]==n):
			self.C = np.transpose(np.array(C))
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
			return

	def setx0(self,x0):
		if(np.shape(x0)[0]==np.shape(self.A)[0]):
			self.x0 = x0
		else:
			print('x0 dimensions should be (',np.shape(self.A)[0],',), please try again.')
			return

	def set_plot_points(self,points):
		if(points<10000):
			self.plot_points = points
		return

	#add integral for B when reqd.
	def get_x(self,t):
		x = np.matmul(linalg.expm(self.A*t),self.x0)
		return x

	def get_y(self,t):
		y = np.matmul(np.transpose(self.C),self.get_x(t))
		return y

	def get_C_dim(self):
		dim = np.shape(self.C)
		return dim[1]

	def plot(self,times):
		start,end = times
		points = self.plot_points
		t = np.linspace(start,end,points)
		x = np.zeros((len(t),len(self.x0)))
		y = np.zeros((len(t),self.get_C_dim()))
		for i,time in enumerate(t):
			x[i,:] = self.get_x(time)
			y[i,:] = np.matmul(self.C,self.get_x(time))
		labels_x = ["x"+str(i) for i in range(0,len(self.x0))]
		labels_y = ["y"+str(i) for i in range(0,len(self.x0))]
		f, axarr = plt.subplots(2,sharex=True)
		for x_arr,x_label in zip(x.transpose(),labels_x):
			axarr[0].plot(t,x_arr,label = x_label)
		for y_arr,y_label in zip(y.transpose(),labels_y):
			axarr[1].plot(t,y_arr,label = y_label)
		axarr[0].set_title('Internal value plot for t = '+str(start)+' to t = '+str(end)+'.')
		axarr[1].set_title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
		plt.xlabel('Time')
		axarr[0].legend()
		axarr[1].legend()
		plt.show()



	def plot_x(self,times):
		start,end = times
		points = self.plot_points
		t = np.linspace(start,end,points)
		x = np.zeros((len(t),len(self.x0)))
		for i,time in enumerate(t):
			x[i,:] = self.get_x(time)
		labels = ["x"+str(i) for i in range(0,len(self.x0))]
		plt.xlabel('Time')
		plt.title('Internal value plot for t = '+str(start)+' to t = '+str(end)+'.')
		for x_arr,label in zip(x.transpose(),labels):
			plt.plot(t,x_arr,label = label)
		plt.legend()
		plt.show()

	def plot_y(self,times):
		start,end = times
		points = self.plot_points
		t = np.linspace(start,end,points)
		y = np.zeros((len(t),self.get_C_dim()))
		for i,time in enumerate(t):
			y[i,:] = np.matmul(self.C,self.get_x(time))
		labels = ["y"+str(i) for i in range(0,len(self.x0))]
		plt.xlabel('Time')
		plt.title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
		for y_arr,label in zip(y.transpose(),labels):
			plt.plot(t,y_arr,label = label)
		plt.legend()
		plt.show()



