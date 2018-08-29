# Features 
# sum_of_neighbors: Sum of Neighbors
# common_neighbors: Sum of Common Neighbors
# shortest_distance: Shortest Distance
# katz_centrality: Katz Centrality

import math
import networkx as nx
from scipy import sparse
import numpy as np

def main():
    # generate directed graph from training dataset
    # G = nx.read_adjlist("train.txt", create_using=nx.DiGraph())
    # print(list(G.successors('4066935')))
    # print(list(G.successors('3676114')))
    # phi = (1+math.sqrt(5))/2.0
    # centrality = nx.katz_centrality(G,1/phi-0.01)
    # print(centrality('4066935'))
    return None

if __name__ == "__main__":
    main()