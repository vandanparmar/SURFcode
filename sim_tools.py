import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from graph_tools import *

def random_mat(a,b):
	toReturn = np.random.rand(a,b)
	return toReturn

def random_stable(n):
	toReturn = np.random.rand(n,n)*2-1
	while (np.real(linalg.eigvals(toReturn))<=0).sum() != n:
		toReturn = np.random.rand(n,n)*2-1
	return toReturn

def random_unit(n):
	toReturn = np.random.rand(n,n)*2-1
	if((np.abs(linalg.eigvals(toReturn))<=1).sum() != n):
		max_e = np.max(np.abs(linalg.eigvals(toReturn)))
		toReturn /= max_e*np.random.uniform(1,5)
	return toReturn

class simulate_cont:
	def __init__(self,n=None,no=None,nu=None):
		if(n == None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = None
			self.__ready = False
		else:
			self.__ready = True
			self.A = random_stable(n)
			if(no == None):
				self.C = None
			else:
				self.C = random_mat(n,no)
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

		if(C != None):
			if(np.shape(C)[0]==n):
				self.C = np.array(C)
			elif(np.shape(C)[1]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')

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
		return self

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
		return self

	def setB(self,B):
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
		n = np.shape(self.A)[0]
		if(np.shape(C)[0]==n):
			self.C = np.array(C)
		elif(np.shape(C)[1]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		return self

	def setx0(self,x0):
		if(np.shape(x0)[0]==np.shape(self.A)[0]):
			self.x0 = x0
		else:
			print('x0 dimensions should be (',np.shape(self.A)[0],',), please try again.')
		return self

	def set_plot_points(self,points):
		if(points<10000):
			self.plot_points = points
		return self

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
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[1]
			return toReturn

	def get_x_set(self,times):
		if(self.ready()):
			xs = np.zeros((len(times),len(self.x0)))
			for i,time in enumerate(times):
				xs[i,:] = self.get_x(time)
			return xs

	def get_y_set(self,times,xs=None):
		if(self.ready()):
			if(xs==None):
				ys = np.zeros((len(times),self.get_C_dim()))
				for i,time in enumerate(times):
					ys[i,:] = self.get_y(time)
			else:
				ys = np.matmul(np.transpose(self.C),np.transpose(xs))		
				ys = np.transpose(ys)
		return ys

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
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,xs,header=comment)	
		return self

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
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart time: '+str(start)+'\nend time: '+str(end)
				np.savetxt(filename,ys,header=comment)			
		return self

	def plot(self,times,plot_points=None,filename=None,grid=False):
		if(self.ready()):
			if(self.C == None):
				self.plot_state(times,plot_points,filename,grid)
				return
			if(plot_points==None):
				plot_points = self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			x = self.get_x_set(t)
			y = self.get_y_set(t,x)
			labels_x = ["x"+str(i) for i in range(0,len(self.x0))]
			labels_y = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.subplot(2,1,1)
			for x_arr,x_label in zip(x.transpose(),labels_x):
				plt.plot(t,x_arr,label = x_label)
			plt.title('State plot for t = '+str(start)+' to t = '+str(end)+'.')
			plt.ylabel('State')
			plt.xlabel('Time')
			plt.legend()
			plt.subplot(2,1,2)
			for y_arr,y_label in zip(y.transpose(),labels_y):
				plt.plot(t,y_arr,label = y_label)
			plt.title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
			plt.ylabel('Output')
			plt.xlabel('Time')
			plt.legend()
			if(grid):
				plt.subplot(2,1,1)
				plt.grid(color = '#a6a5a6')
				plt.subplot(2,1,2)
				plt.grid(color = '#a6a5a6')
			plt.subplots_adjust(hspace=0.5)
			plt.show()
			if(filename != None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,times,points,x)
				self.save_output(filename_y,times,points,y)

	def plot_state(self,times,plot_points=None,filename=None,grid=False):
		if(self.ready()):
			if(plot_points==None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			x = self.get_x_set(t)
			labels = ["x"+str(i) for i in range(0,len(self.x0))]
			plt.xlabel('Time')
			plt.ylabel('State')
			plt.title('State plot for t = '+str(start)+' to t = '+str(end)+'.')
			for x_arr,label in zip(x.transpose(),labels):
				plt.plot(t,x_arr,label = label)
			plt.legend()
			if(grid):
				plt.grid(color = '#a6a5a6')
			plt.show()
			if(filename != None):
				self.save_state(filename_x,times,points,x)

	def plot_output(self,times,plot_points=None,filename=None,grid=False):
		if(self.ready()):
			if(plot_points==None):
				plot_points=self.plot_points
			start,end = times
			points = plot_points
			t = np.linspace(start,end,points)
			y = self.get_y_set(t)
			labels = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.xlabel('Time')
			plt.ylabel('Output')
			plt.title('Output plot for t = '+str(start)+' to t = '+str(end)+'.')
			for y_arr,label in zip(y.transpose(),labels):
				plt.plot(t,y_arr,label = label)
			plt.legend()
			if(grid):
				plt.grid(color = '#a6a5a6')
			plt.show()
			if(filename != None):
				self.save_output(filename_y,times,points,y)



class simulate_disc:
	def __init__(self, n=None, no=None, nu=None):
		if(n==None):
			print('Initilising with empty matrices, please specify using "setABC".')
			self.A = np.array([])
			self.B = None
			self.C = None
			self.__ready = False
		else:
			self.__ready = True
			self.A = random_unit(n)
			if(no == None):
				self.C = None
			else:
				self.C = random_mat(n,no)
			if(nu == None):
				self.B = None
			else:
				self.B = random_mat(n,nu)
			self.x0 = np.random.rand(n)			

	def setABC(self,A,C=None,B=None):
		shapeA = np.shape(A)
		if(shapeA[0] == shapeA[1]):
			self.A = np.array(A)
			n = shapeA[0]
			self.x0 = np.random.rand(n)
			self.__ready = True
		else:
			print('Please supply a square A matrix.')
		
		if(C != None):
			if(np.shape(C)[0]==n):
				self.C = np.array(C)
			elif(np.shape(C)[1]==n):
			#	self.C = np.transpose(np.array(C))
				print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
			else:
				print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		
		if(B != None):
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
		if(self.C != None):
			if(np.shape(A)[0]==np.shape(self.C)[0]):
				self.A = np.array(A)
				self.x0 = np.random.rand(np.shape(A)[0])
			else:
				print('Dimensions of A not compatible, please try again.')
		else:
			print('Please set A, B and C using setABC.')
		return self

	def setB(self,B):
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
		n = np.shape(self.A)[0]
		if(np.shape(C)[0]==n):
			self.C = np.array(C)
		elif(np.shape(C)[1]==n):
		#	self.C = np.transpose(np.array(C))
			print('Dimensions ',np.shape(C),' are not acceptable. You may wish to transpose this matrix.')
		else:
			print('Dimensions ',np.shape(C),' are not acceptable, please reenter.')
		return self

	def setx0(self,x0):
		if(np.shape(x0)[0]==np.shape(self.A)[0]):
			self.x0 = x0
		else:
			print('x0 dimensions should be (',np.shape(self.A)[0],',), please try again.')
		return self

	def get_x(self,k):
		if(self.ready()):
			x = np.matmul(np.linalg.matrix_power(self.A,k),self.x0)
			return x

	def get_y(self,k):
		if(self.ready()):
			y = np.matmul(np.transpose(self.C),self.get_x(k))
			return y

	def get_x_set(self,ks):
		if(self.ready()):
			xs = np.zeros((len(ks),len(self.x0)))
			xs[0,:] = self.get_x(ks[0])
			for i,time in enumerate(ks[1:]):
				xs[i+1,:] = np.matmul(self.A,xs[i,:])
		return xs

	def get_y_set(self,ks,xs=None):
		if(self.ready()):
			if(xs==None):
				ys = np.zeros((len(ks),self.get_C_dim()))
				x_0 = self.get_x(ks[0])
				ys[0,:] = np.matmul(np.transpose(self.C),x_0)
				for i,time in enumerate(ks[1:]):
					x_0 = np.matmul(self.A,x_0)
					ys[i+1,:] = np.matmul(np.transpose(self.C),x_0)
			else:
				ys = np.matmul(np.transpose(self.C),np.transpose(xs))		
				ys = np.transpose(ys)
		return ys

	def get_C_dim(self):
		if(self.ready()):
			dim = np.shape(self.C)
			if(len(dim)==1):
				toReturn = 1
			else:
				toReturn = dim[1]
			return toReturn

	def save_state(self,filename,ks,xs=None):
		if(self.ready()):
			eigvals = linalg.eigvals(self.A)
			start,end = ks
			if(xs==None):
				k = np.arange(start,end)
				xs = np.zeros((len(k),len(self.x0)))
				xs[0,:] = self.get_x(start)
				for i,time in enumerate(k[1:]):
					xs[i+1,:] = np.matmul(self.A,xs[i,:])
			if(len(xs)>10000):
				print('Too many states to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,xs,header=comment)
		return self

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
					x_0 = np.matmul(self.A,x_0)
					ys[i+1,:] = np.matmul(np.transpose(self.C),x_0)
			if(len(ys)>10000):
				print('Too many outputs to save.')
			else:
				comment = 'A eigenvalues: '+ str(eigvals)+'\nstart k: '+str(start)+'\nend k: '+str(end)
				np.savetxt(filename,ys,header=comment)
		return self

	def plot(self,ks,filename=None, grid=False):
		if(self.ready()):
			if(self.C == None):
				self.plot_state(ks,filename,grid)
				return
			start,end = ks
			k = np.arange(start,end)
			x = self.get_x_set(k)
			y = self.get_y_set(k,x)
			labels_x = ["x"+str(i) for i in range(0,len(self.x0))]
			labels_y = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.subplot(2,1,1)
			for x_arr,x_label in zip(x.transpose(),labels_x):
				plt.step(k,x_arr,where='post',label = x_label)
			plt.title('State plot for k = '+str(start)+' to k = '+str(end)+'.')
			plt.ylabel('State')
			plt.xlabel('k')
			plt.legend()
			plt.subplot(2,1,2)
			for y_arr,y_label in zip(y.transpose(),labels_y):
				plt.step(k,y_arr,where='post',label = y_label)
			plt.title('Output plot for k = '+str(start)+' to k = '+str(end)+'.')
			plt.ylabel('Output')
			plt.xlabel('k')
			plt.legend()
			if(grid):
				plt.subplot(2,1,1)
				plt.grid(color = '#a6a5a6')
				plt.subplot(2,1,2)
				plt.grid(color = '#a6a5a6')
			plt.subplots_adjust(hspace = 0.5)
			plt.show()
			if(filename!=None):
				filename_x = 'state_'+filename
				filename_y = 'output_'+filename
				self.save_state(filename_x,ks,x)
				self.save_output(filename_y,ks,y)

	def plot_state(self,ks,filename=None, grid=False):
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end)
			x = self.get_x_set(k)
			labels = ["x"+str(i) for i in range(0,len(self.x0))]
			plt.xlabel('k')
			plt.ylabel('State')
			plt.title('State plot for k = '+str(start)+' to k = '+str(end)+'.')
			for x_arr,label in zip(x.transpose(),labels):
				plt.step(k,x_arr,where='post',label = label)
			plt.legend()
			if(grid):
				plt.grid(color = '#a6a5a6')
			plt.show()
			if(filename!=None):
				self.save_state(filename,ks,x)

	def plot_output(self,ks,filename=None, grid=False):
		if(self.ready()):
			start,end = ks
			k = np.arange(start,end)
			y = self.get_y_set(k)
			labels = ["y"+str(i) for i in range(0,self.get_C_dim())]
			plt.xlabel('k')
			plt.ylabel('Output')
			plt.title('Output plot for k = '+str(start)+' to k = '+str(end)+'.')
			for y_arr,label in zip(y.transpose(),labels):
				plt.step(k,y_arr,where='post',label = label)
			plt.legend()
			if(grid):
				plt.grid(color = '#a6a5a6')
			plt.show()
			if(filename!=None):
				self.save_state(filename,ks,y)

