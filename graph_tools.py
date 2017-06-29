import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_adj_matrix(n):
    G = nx.binomial_graph(n,random.random())
    return nx.to_numpy_matrix(G)


def show_graph(adjacency_matrix):
    # given an adjacency matrix use networkx and matlpotlib to plot the graph
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    # nx.draw(gr) # edited to include labels
    nx.draw_networkx(gr)
    plt.show() 