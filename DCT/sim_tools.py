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
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt

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
