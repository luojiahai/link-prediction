import math
import networkx as nx
from scipy import sparse
import numpy as np
from sklearn.svm import SVC

def train_svm(train_pos, train_neg, network):
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    X_pos = []
    X_neg = []

    return None

def test_svm(test_data, network):
    return None

def feature_extraction(link, network):
    feature = []
    (x, y) = link

    sample = enclosing_subgraph_extraction((x, y), network, 2)
    subgraph = nx.DiGraph()
    for elem in sample:
        subgraph.add_edge(elem[0], elem[1])
    
    # common neighbors
    common_neighbors = len(list(nx.common_neighbors(subgraph.to_undirected(), x, y)))
    feature.append(common_neighbors)
    # jaccard coefficient
    (_, _, jaccard_coefficient) = nx.jaccard_coefficient(subgraph.to_undirected(), [(x, y)])
    feature.append(jaccard_coefficient)
    # preferential attachment
    (_, _, preferential_attachment) = nx.preferential_attachment(subgraph.to_undirected(), [(x, y)])
    feature.append(preferential_attachment)
    # adamic adar index
    (_, _, adamic_adar_index) = nx.adamic_adar_index(subgraph.to_undirected(), [(x, y)])
    feature.append(adamic_adar_index)
    # resource allocation index
    (_, _, resource_allocation_index) = nx.resource_allocation_index(subgraph.to_undirected(), [(x, y)])
    feature.append(resource_allocation_index)
    
    return feature

def list_complement(list1, list2):
    return [elem for elem in list1 if elem not in list2]

def list_union(list1, list2):
    output = list1
    for elem in list2:
        if elem not in output:
            output.append(elem)
    return output

def neighbors(fringe, network):
    output = []
    for i in range(len(fringe)):
        ind = fringe[i]
        x = ind[0]
        y = ind[1]
        xy = network.getrow(x).nonzero()[1]
        yx = network.getcol(y).nonzero()[0]
        for elem in xy:
            link = [x, elem]
            if link not in output:
                output.append(link)
        for elem in yx:
            link = [elem, y]
            if link not in output:
                output.append(link)
    return output

def enclosing_subgraph_extraction(link, network, h):
    (x, y) = link
    fringe = links = [[x, y]]
    for dist in range(h):
        if len(fringe) == 0:
            break
        fringe = neighbors(fringe, network)
        fringe = list_complement(fringe, links)
        links = list_union(links, fringe)
    sample = np.array(links)
    return sample

def sample_negative_links(size):
    return None

def load_test_data(path, delimiter):
    test = []
    with open(path, "r") as f:
        for line in f:
            # split the line
            splited = line.rstrip().split(delimiter)
            for i in range(1, len(splited)):
                # two connected vertices
                v1 = splited[0]
                v2 = splited[i]
                # construct (x, y) tuple
                test.append((v1, v2))
    return test

def load_train_data(path, delimiter):
    train = []
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
                # construct (x, y) tuple
                train.append((v1, v2))
    
    # total number of distinct nodes
    n = len(adjlist.keys())
    # row indices
    row_ind = np.array(row_array)
    # column indices
    col_ind = np.array(col_array)
    # data to be stored in COO sparse matrix
    data = np.ones(len(row_array))
    sparse_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n,n), dtype=np.uint8)
    
    return (train, adjlist, sparse_matrix)

def main():
    (train, adjlist, sparse_matrix) = load_train_data("data/Celegans.txt", delimiter=' ')
    test = load_test_data("data/Celegans_test.txt", delimiter=' ')
    negative_links = sample_negative_links(size=len(train))

    print(feature_extraction((0,6), sparse_matrix))

    


if __name__ == "__main__":
    main()
