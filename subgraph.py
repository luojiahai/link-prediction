import math
from scipy import sparse
import numpy as np

def list_union(list1, list2):
    output = []
    for elem in list1:
        if elem not in output:
            output.append(elem)
    for elem in list2:
        if elem not in output:
            output.append(elem)
    return output

def neighbors(fringe, network):
    for i in range(len(fringe)):
        ind = fringe[i]
        x = ind[0]
        y = ind[1]
        x_row = network.getrow(x).nonzero()[1]
        y_col = network.getcol(y).nonzero()[0]
    return None

def enclosing_subgraph_extraction(link, network, h):
    (x, y) = link
    fringe = links = [[x, y]]
    for dist in range(h):
        if len(fringe) == 0:
            break
        fringe = neighbors(fringe, network)
        links = union(links, fringe)
    sample = np.array(links)
    return sample

def sample_negative_links():
    return None

def load_data(path, delimiter):
    adjlist = {}    # adjacency list
    row_array = []  # row indices
    col_array = []  # col indices

    with open(path, "r") as f:
        for line in f:
            # split the line
            splited = line.rstrip().split(delimiter)
            for i in range(1, len(splited)):
                # two connected vertices
                v1 = splited[0]
                v2 = splited[i]
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
    (adjlist, sparse_matrix) = load_data("data/Celegans.txt", delimiter=' ')
    print(sparse_matrix)
    # enclosing_subgraph_extraction((284, 277), sparse_matrix, 1)
    a = sparse_matrix.getcol(3).nonzero()[0]
    print(a)

if __name__ == "__main__":
    main()
