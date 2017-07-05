import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json


#generate a random graph with a given number of nodes and probability for each edge.
def generate_adj_matrix(n,p):
    G = nx.connected_watts_strogatz_graph(n,3,p)
    return nx.to_numpy_matrix(G)



# given an adjacency matrix use networkx and matlpotlib to plot the graph
def show_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    # nx.draw(gr) # edited to include labels
    nx.draw_spring(gr)
    plt.show() 


def save_to_file(adjacency_matrix,filename):
	np.savetxt(filename,adjacency_matrix)
	print('File saved!')


def load_from_file(filename):
	if filename.endswith('.json'):
		toReturn = load_from_GUI(filename)
	if filename.endswith(('.txt','.dat')):
		toReturn = load_from_numpy(filename)
	return toReturn


def load_from_GUI(filename):
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
	toReturn = np.loadtxt(filename)
	return toReturn

#save_to_file(load_from_GUI('test_graph_1.json'),'test_graph_2.txt')
#show_graph(load_from_file('test_graph_2.txt'))
#show_graph(load_from_file('test_graph_1.json'))