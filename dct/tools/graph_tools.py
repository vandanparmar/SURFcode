"""
Graph Tools (:mod:`graph_tools`)
==============================================

This module contains network related functions.

Generating Matrices
*******************

.. autosummary::
	:toctree:

	generate_rand
	generate_laplacian
	generate_degree
	generate_incidence


Displacing and Importing Matrices
*********************************

.. autosummary::
	:toctree:

	show_graph
	load_from_file
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json


__all__ = ["generate_rand","generate_laplacian","generate_degree","generate_incidence","show_graph","save_to_file","load_from_file"]

#generate a random graph with a given number of nodes and probability for each edge.
def generate_rand(n,p):
	"""Summary
	
	Args:
	    n (TYPE): Description
	    p (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	G = nx.connected_watts_strogatz_graph(n,3,p)
	toReturn = nx.to_numpy_matrix(G)
	return toReturn

def generate_laplacian(adjacency_matrix):
	"""Summary
	
	Args:
	    adjacency_matrix (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	G = nx.from_numpy_matrix(adjacency_matrix)
	toReturn = nx.laplacian_matrix(G)
	return toReturn.toarray()

def generate_degree(adjacency_matrix):
	"""Summary
	
	Args:
	    adjacency_matrix (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	laplacian = generate_laplacian(adjacency_matrix)
	toReturn = laplacian + adjacency_matrix
	return toReturn

def generate_incidence(adjacency_matrix):
	"""Summary
	
	Args:
	    adjacency_matrix (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	G = nx.from_numpy_matrix(adjacency_matrix)
	toReturn = nx.incidence_matrix(G)
	return toReturn.toarray()

# given an adjacency matrix use networkx and matlpotlib to plot the graph
def show_graph(adjacency_matrix):
	"""Summary
	
	Args:
	    adjacency_matrix (TYPE): Description
	"""
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	gr = nx.Graph()
	gr.add_edges_from(edges)
	# nx.draw(gr) # edited to include labels
	mapping=dict(zip(gr.nodes(),np.arange(1,len(gr.nodes())+1)))
	gr = nx.relabel_nodes(gr,mapping)
	nx.draw_spring(gr,with_labels=True,font_color='w')
	plt.show() 

def save_to_file(adjacency_matrix,filename):
	"""Summary
	
	Args:
	    adjacency_matrix (TYPE): Description
	    filename (TYPE): Description
	"""
	np.savetxt(filename,adjacency_matrix)
	print('File saved!')

def load_from_file(filename):
	"""Summary
	
	Args:
	    filename (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	if filename.endswith('.json'):
		toReturn = load_from_GUI(filename)
	if filename.endswith(('.txt','.dat')):
		toReturn = load_from_numpy(filename)
	return toReturn

#load file from the online GUI
def load_from_GUI(filename):
	"""Summary
	
	Args:
	    filename (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	with open(filename) as data_file:    
		data = json.load(data_file)
		nodes_num = len(data['nodes'])
		toReturn = np.zeros((nodes_num,nodes_num))
		nodes_map = {}
		for i,node in enumerate(data['nodes']):
			nodes_map[node] = i
		for edge in data['edges']:
			toReturn[nodes_map[edge['source']]][nodes_map[edge['target']]]+=1
		toReturn += np.transpose(toReturn)
		toReturn = np.clip(toReturn,0.0,1.0)
		return toReturn

def load_from_numpy(filename):
	"""Summary
	
	Args:
	    filename (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	toReturn = np.loadtxt(filename)
	return toReturn
