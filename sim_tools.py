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

def random_unit(n):
	toReturn = np.random.rand(n,n)*2-1
	if((np.abs(linalg.eigvals(toReturn))<=1).sum() != n):
		max_e = np.max(np.abs(linalg.eigvals(toReturn)))
		toReturn /= max_e*np.random.uniform(1,100)
	return toReturn

class simulate_cont:
	def __init__(self,n=None,n0=None,nu=None):
		if(n == None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = np.array([])
			self.__ready = False
		else:
			self.__ready = True
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
			self.__ready = True
		else:
			print('Please supply a square A matrix.')
			return

		if(C != None):
			if(np.shape(C)[0]==n):
				self.C = np.array(C)
			elif(np.shape(C)[1]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
				return

		if(self.C==None):
			self.C = np.identity(n)

				
		if(B != None):
			if(np.shape(B)[0]==n):
				self.B = np.array(B)
			elif(np.shape(B)[1]==n):
			#	self.B = np.transpose(np.array(B))
				print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
				return

	def ready(self):
		if(self.__ready):
			return True
		else:
			print('Please set A, B and C using setABC.')
			return False

	def setA(self,A):
		if(self.C != None):
			if(np.shape(A)[0]==np.shape(self.C)[0]):
				self.A = np.array(A)
				self.x0 = np.random.rand(np.shape(A)[0])
			else:
				print('Dimensions of A not compatible, please try again.')
		else:
			print('Please set A, B and C using setABC.')
		return

	def setB(self,B):
		n = np.shape(self.A)[0]
		if(np.shape(B)[0]==n):
			self.B = np.array(B)
		elif(np.shape(B)[1]==n):
		#	self.B = np.transpose(np.array(B))
			print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
			return

	def setC(self,C):
		n = np.shape(self.A)[0]
		if(np.shape(C)[0]==n):
			self.C = np.array(C)
		elif(np.shape(C)[1]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
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
		if(self.ready()):
			x = np.matmul(linalg.expm(self.A*t),self.x0)
			return x

	def get_y(self,t):
		if(self.ready()):
			y = np.matmul(np.transpose(self.C),self.get_x(t))
			return y

	def get_C_dim(self):
		if(self.ready()):
			dim = np.shape(self.C)
			return dim[1]

	def save_state(self,filename,times,plot_points=None,xs=None):	
		if(self.ready()):
			if(plot_points==None):
				plot_points = self.plot_points
			eigvals = linalg.eigvals(self.A)
			start,end = times
			if(xs == None):
				t = np.linspace(start,end,plot_points)
				xs = np.zeros((len(t),len(self.x0)))
				for i,time in enumerate(t):
					xs[i,:] = self.get_x(time)
			if(len(xs)>10000):
				print('Too many states to save.')
				return
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,xs,header=comment)	

	def save_output(self,filename,times,plot_points=None,ys=None):
		if(self.ready()):
			if(plot_points==None):
				plot_points = self.plot_points
			eigvals = linalg.eigvals(self.A)
			start,end = times
			if(ys == None):
				t = np.linspace(start,end,plot_points)
				ys = np.zeros((len(t),self.get_C_dim()))
				for i,time in enumerate(t):
					ys[i,:] = self.get_y(time)
			if(len(ys)>10000):
				print('Too many outputs to save.')
				return
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,ys,header=comment)			

	def plot(self,times,plot_points=None,filename=None):
		if(self.ready()):
			if(plot_points==None):
				plot_points = self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			x = np.zeros((len(t),len(self.x0)))
			y = np.zeros((len(t),self.get_C_dim()))
			for i,time in enumerate(t):
				x[i,:] = self.get_x(time)
				y[i,:] = self.get_y(time)
			labels_x = ["x"+str(i) for i in range(0,len(self.x0))]
			labels_y = ["y"+str(i) for i in range(0,self.get_C_dim())]
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
			if(filename != None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,times,points,x)
				self.save_output(filename_y,times,points,y)

	def plot_x(self,times,plot_points=None,filename=None):
		if(self.ready()):
			if(plot_points==None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
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
			if(filename != None):
				self.save_state(filename_x,times,points,x)

	def plot_y(self,times,plot_points=None,filename=None):
		if(self.ready()):
			if(plot_points==None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			y = np.zeros((len(t),self.get_C_dim()))
			for i,time in enumerate(t):
				y[i,:] = np.matmul(np.transpose(self.C),self.get_x(time))
			labels = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.xlabel('Time')
			plt.title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
			for y_arr,label in zip(y.transpose(),labels):
				plt.plot(t,y_arr,label = label)
			plt.legend()
			plt.show()
			if(filename != None):
				self.save_output(filename_y,times,points,y)



class simulate_disc:
	def __init__(self, n=None, n0=None, nu=None):
		if(n==None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = np.array([])
			self.__ready = False
		else:
			self.__ready = True
			self.A = random_unit(n)
			if(n0 == None):
				self.C = np.identity(n)
			else:
				self.C = random_mat(n,n0)
			if(nu == None):
				self.B = None
			else:
				self.B = random_mat(n,nu)
			self.x0 = np.random.rand(n)			
		return

	def setABC(self,A,C=None,B=None):
		shapeA = np.shape(A)
		if(shapeA[0] == shapeA[1]):
			self.A = np.array(A)
			n = shapeA[0]
			self.x0 = np.random.rand(n)
			self.__ready = True
		else:
			print('Please supply a square A matrix.')
			return

		if(C != None):
			if(np.shape(C)[0]==n):
				self.C = np.array(C)
			elif(np.shape(C)[1]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
				return
		if(self.C==None):
			self.C = np.identity(n)
				
		if(B != None):
			if(np.shape(B)[0]==n):
				self.B = np.array(B)
			elif(np.shape(B)[1]==n):
			#	self.B = np.transpose(np.array(B))
				print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
				return

	def ready(self):
		if(self.__ready):
			return True
		else:
			print('Please set A, B and C using setABC.')
			return False

	def setA(self,A):
		if(self.C != None):
			if(np.shape(A)[0]==np.shape(self.C)[0]):
				self.A = np.array(A)
				self.x0 = np.random.rand(np.shape(A)[0])
			else:
				print('Dimensions of A not compatible, please try again.')
		else:
			print('Please set A, B and C using setABC.')
		return

	def setB(self,B):
		n = np.shape(self.A)[0]
		if(np.shape(B)[0]==n):
			self.B = np.array(B)
		elif(np.shape(B)[1]==n):
		#	self.B = np.transpose(np.array(B))
			print('Dimensions ',np.shape(B),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(B),' are not acceptable, please reenter.')
			return

	def setC(self,C):
		n = np.shape(self.A)[0]
		if(np.shape(C)[0]==n):
			self.C = np.array(C)
		elif(np.shape(C)[1]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
			return

	def setx0(self,x0):
		if(np.shape(x0)[0]==np.shape(self.A)[0]):
			self.x0 = x0
		else:
			print('x0 dimensions should be (',np.shape(self.A)[0],',), please try again.')
			return

	def get_x(self,k):
		if(self.ready()):
			x = np.matmul(np.linalg.matrix_power(self.A,k),self.x0)
			return x

	def get_y(self,k):
		if(self.ready()):
			y = np.matmul(np.transpose(self.C),self.get_x(k))
			return y

	def get_C_dim(self):
		if(self.ready()):
			dim=np.shape(self.C)
			return dim[1]		

	def save_state(self,filename,ks,xs=None):
		if(self.ready()):
			eigvals = linalg.eigvals(self.A)
			start,end = ks
			if(xs==None):
				k = np.arange(start,end)
				xs = np.zeros((len(k),len(self.x0)))
				xs[0,:] = self.get_x(start)
				for i,time in enumerate(k[1:]):
					xs[i+1,:] = np.matmul(self.A,x[i,:])
			if(len(xs)>10000):
				print('Too many states to save.')
				return
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,xs,header=comment)

	def save_output(self,filename,ks,ys=None):
		if(self.ready()):
			eigvals = linalg.eigvals(self.A)
			start,end = ks
			if(ys==None):
				k = np.arange(start,end)
				ys = np.zeros((len(k),self.get_C_dim()))
				x_0 = self.get_x(start)
				ys[0,:] = np.matmul(np.transpose(self.C),x_0)
				for i,time in enumerate(k[1:]):
					x_0 = np.matmul(self.A,xs)
					ys[i+1,:] = np.matmul(np.transpose(self.C),x_0)
			if(len(xs)>10000):
				print('Too many outputs to save.')
				return
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,ys,header=comment)

	def plot(self,ks,filename=None):
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end)
			x = np.zeros((len(k),len(self.x0)))
			y = np.zeros((len(k),self.get_C_dim()))
			x[0,:] = self.get_x(start)
			C = np.transpose(self.C)
			y[0,:] = np.matmul(C,x[0,:])
			for i,time in enumerate(k[1:]):
				x[i+1,:] = np.matmul(self.A,x[i,:])
				y[i+1,:] = np.matmul(np.transpose(self.C),x[i+1,:])
			labels_x = ["x"+str(i) for i in range(0,len(self.x0))]
			labels_y = ["y"+str(i) for i in range(0,self.get_C_dim())]
			f, axarr = plt.subplots(2,sharex=True)
			for x_arr,x_label in zip(x.transpose(),labels_x):
				axarr[0].step(k,x_arr,where='post',label = x_label)
			for y_arr,y_label in zip(y.transpose(),labels_y):
				axarr[1].step(k,y_arr,where='post',label = y_label)
			axarr[0].set_title('Internal value plot for t = '+str(start)+' to t = '+str(end)+'.')
			axarr[1].set_title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
			plt.xlabel('Time')
			axarr[0].legend()
			axarr[1].legend()
			plt.show()
			if(filename!=None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,ks,x)
				self.save_output(filename_y,ks,y)

	def plot_x(self,ks,filename=None):
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end)
			x = np.zeros((len(k),len(self.x0)))
			x[0,:] = self.get_x(start)
			for i,time in enumerate(k[1:]):
				x[i+1,:] = np.matmul(self.A,x[i,:])
			labels = ["x"+str(i) for i in range(0,len(self.x0))]
			plt.xlabel('Time')
			plt.title('Internal value plot for t = '+str(start)+' to t = '+str(end)+'.')
			for x_arr,label in zip(x.transpose(),labels):
				plt.step(k,x_arr,where='post',label = label)
			plt.legend()
			plt.show()
			if(filename!=None):
				self.save_state(filename,ks,x)

	def plot_y(self,ks,filename=None):
		if(self.ready()):
			start,end = ks
			points = self.plot_points
			k = np.arange(start,end)
			y = np.zeros((len(k),self.get_C_dim()))
			x_0 = self.get_x(start)
			y[0,:] = np.matmul(np.transpose(self.C),x_0)
			for i,time in enumerate(k[1,:]):
				x_0 = np.matmul(self.A,x_0)
				y[i+1,:] = np.matmul(np.transpose(self.C),x_0)
			labels = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.xlabel('Time')
			plt.title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
			for y_arr,label in zip(y.transpose(),labels):
				plt.step(k,y_arr,where='post',label = label)
			plt.legend()
			plt.show()
			if(filename!=None):
				self.save_state(filename,ks,y)