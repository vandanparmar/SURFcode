import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
    nx.draw(gr)
    plt.show() 

