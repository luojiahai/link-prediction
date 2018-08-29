# Features 
# sum_of_neighbors: Sum of Neighbors
# common_neighbors: Sum of Common Neighbors
# shortest_distance: Shortest Distance
# katz_centrality: Katz Centrality

import math
from scipy import sparse
import numpy as np

def neighbors(fringe, network):
    new_fringe = []


def enclosing_subgraph_extraction(link, network, h):
    (i, j) = link
    fringe = links = [i, j]
    nodes_dist = [0, 0]
    print(fringe)
    print(links)
    for dist in range(h):
        print("OK")
    return None

def load_data(path):
    adjlist = {}    # adjacency list
    row_array = []  # row indices
    col_array = []  # col indices

    with open(path, "r") as f:
        for line in f:
            # split the line
            splited = line.rstrip().split(' ')
            # two connected vertices
            v1 = splited[0]
            v2 = splited[1]
            # construct adjacency list
            if v1 not in adjlist.keys():
                adjlist[v1] = [v2]
            else:
                adjlist[v1].append(v2)
            if v2 not in adjlist.keys():
                adjlist[v2] = []
            # construct row and col indices arrays
            row_array.append(v1)
            col_array.append(v2)

    # total number of distinct nodes
    n = len(adjlist.keys())
    # row indices
    row_ind = np.array(row_array)
    # column indices
    col_ind = np.array(col_array)
    # data to be stored in COO sparse matrix
    data = np.ones(len(row_array))
    sparse_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n,n), dtype=np.uint8)
    
    return (adjlist, sparse_matrix)

def main():
    (adjlist, sparse_matrix) = load_data("data/Celegans.txt")
    print(sparse_matrix)
    enclosing_subgraph_extraction((284, 277), sparse_matrix, 1)

if __name__ == "__main__":
    main()
