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


"""

import numpy as np
from scipy import linalg,stats
import matplotlib.pyplot as plt
from matplotlib import colors,cm

__all__ = ["random_mat","random_stable","random_unit","plot_sio","plot_resp","compass","marg_stab","plot_hmap"]

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

def random_stable_2():
	A = 2*random_mat(2,2)-1
	while (np.real(linalg.eigvals(A))<=0).sum() != 2:
		A = 2*random_mat(2,2)-1
	return A

def random_stable(n):
	"""Generate a random n x n stable matrix.
	
	Args:
	    n (int): Dimension of matrix
	
	Returns:
	    ndarray: n x n matrix with random entries uniformly distributed on [0,1), eigenvalues all have negative real part
	"""
	P = stats.special_ortho_group.rvs(n)
	A = np.zeros((n,n))
	i = n
	while i>1:
		A[n-i:n-i+2,n-i:n-i+2] = random_stable_2()
		i -=2
	if(i==1):
		A[n-1,n-1] = -np.random.rand(1)[0]
	return np.matmul(P,np.matmul(A,P.T))

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

def plot_resp(self,times,inputs,outputs,disc,grid,resp,type_i):
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
		plt.title(type_i +" response from input "+str(input_i)+title)
		plt.ylabel("Output")
		plt.xlabel(xlabel)
		plt.legend()
		if(grid):
			plt.grid(color= '#a6a5a6')
	plt.subplots_adjust(hspace=0.5)
	plt.show()

def plot_hmap(self,times,vals,title,xlabel,vmin=1e-10):
	my_cmap = cm.get_cmap('BuPu')
	my_cmap.set_bad((0.9686275,0.9882359411,0.9921568627))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.pcolor(np.absolute(vals), norm=colors.LogNorm(), cmap=my_cmap,vmin=vmin)
	plt.colorbar()
	plt.show()



def compass(pairs, arrowprops=None):
	if (len(pairs)==4):
		it = [(0,0),(0,1),(1,0),(1,1)]
		tot = (2,2)
	elif (len(pairs)==2):
		it = [(0,0),(1,0)]
		tot = (1,2)
	else:
		it = [(0,0)]
		tot = (1,1)
	fig, ax = plt.subplots(tot[0],tot[1],subplot_kw=dict(polar=True))
	for coords,pair in zip(it,pairs):
		if (tot==(2,2)):
			ax_i = ax[coords[0],coords[1]]
		elif (tot==(1,2)):
			ax_i  = ax[coords[0]]
		else:
			ax_i = ax
			to_pair = np.zeros((1))
			to_pair[0] = pair[1]
			pair = (pair[0],to_pair)
		angles = np.angle(pair[1])
		radii = np.abs(pair[1])
		title = "Damping ratio = "+str((pair[0]))
		ax_i.set_title(title)
		def c(angle):
			if((angle<(np.pi/2.0)) and (angle>(-np.pi/2.0))):
				return "#00aa5e" #green
			else:
				return "#e62325" #red
		[ax_i.annotate(i+1, xy=(0,0), xytext=(thing[0],thing[1]), arrowprops=dict(arrowstyle='<-',color=c(thing[0]))) for i,thing in enumerate(zip(angles, radii))]
		ax_i.set_ylim(0, np.max(radii))
		plt.grid(color="#a6a5a6")
		ax_i.set_rlabel_position(270)
	plt.show()

def marg_stab(n):
	"""Generates a chain graph structure with n nodes.

	Args:
		n (int): Number of nodes

	Returns:
		ndarray: Adjacency matrix 
	"""	
	def p(i,j):
		if(i==j):
			return 0.6
		elif(i==j+1):
			return 0.2
		elif(i==j-1):
			return 0.2
		else:
			return 0
	toReturn = [[p(i,j) for i in range(0,n)] for j in range(0,n)]
	toReturn[0][0] = 0.8
	toReturn[n-1][n-1] = 0.8
	return np.array(toReturn)

