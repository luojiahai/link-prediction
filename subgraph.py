import math
import networkx as nx
from scipy import sparse
import numpy as np
from sklearn.svm import SVC
import random
import time
import progressbar


def save_train_feature_vectors(path, train_data, label, network):
    bar = progressbar.ProgressBar(max_value=len(train_data))
    f = open(path, "a")
    i = 0
    for (x, y) in train_data:
        feature = feature_extraction((x, y), network)
        string = str(x) + '\t' + str(y) + '\t' + str(label)
        for elem in feature:
            string += '\t' + str(elem)
        f.write(string + '\n')
        # progress bar update
        i = i + 1
        if (i == len(train_data)):
            time.sleep(0.1)
        bar.update(i)
    print('\n')
    return None
    
def train_svm(path, network):
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    X = []
    y = []
    with open(path, "r") as f:
        for line in f:
            splited = line.rstrip().split('\t')
            # x = int(splited[0])
            # y = int(splited[1])
            label = int(splited[2])
            feature = []
            for i in range(3, len(splited)):
                feature.append(splited[i])
            X.append(feature)
            y.append(label)
    model.fit(X, y)
    return model

def test_svm(model, test_data, test_dict, network):
    testset = []
    with open("test_output.csv", "a") as f:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        f.write("Id,Prediction")
        i = 1
        for (x, y) in test_data:
            feature = feature_extraction((x, y), network)
            results = model.predict_proba([feature])[0]
            prob_per_class_dictionary = dict(zip(model.classes_, results))
            string = str(i) + ',' + str(prob_per_class_dictionary[1])
            f.write(string)
            # progress bar update
            i = i + 1
            bar.update(i)
        print('\n')
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
    (_, _, jaccard_coefficient) = list(nx.jaccard_coefficient(subgraph.to_undirected(), [(x, y)]))[0]
    feature.append(jaccard_coefficient)
    # preferential attachment
    (_, _, preferential_attachment) = list(nx.preferential_attachment(subgraph.to_undirected(), [(x, y)]))[0]
    feature.append(preferential_attachment)
    # adamic adar index
    (_, _, adamic_adar_index) = list(nx.adamic_adar_index(subgraph.to_undirected(), [(x, y)]))[0]
    feature.append(adamic_adar_index)
    # resource allocation index
    (_, _, resource_allocation_index) = list(nx.resource_allocation_index(subgraph.to_undirected(), [(x, y)]))[0]
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

def sample_negative_links(n, size, adjlist):
    bar = progressbar.ProgressBar(max_value=size)
    neg_links = []
    i = 0
    while i < size:
        x = random.randrange(n)
        y = random.randrange(n)
        if (y not in adjlist[x]):
            neg_links.append((x, y))
            i = i + 1
            # progress bar update
            if (i == size):
                time.sleep(0.1)
            bar.update(i)
        else:
            i = i - 1
    print('\n')
    return neg_links

def load_test_data(path, delimiter, with_index):
    test = []
    test_dict = {}
    with open(path, "r") as f:
        for line in f:
            # split the line
            splited = line.rstrip().split(delimiter)
            if (with_index):
                v0 = int(splited[0])
                v1 = int(splited[1])
                v2 = int(splited[2])
                # construct dictionary
                test_dict[v0] = (v1, v2)
                # construct (x, y) tuple
                test.append((v1, v2))
            else:
                v1 = int(splited[0])
                v2 = int(splited[1])
                # construct (x, y) tuple
                test.append((v1, v2))
    return (test, test_dict)

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
                v1 = int(splited[0])
                v2 = int(splited[i])
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
    print("Loading train data...")
    (train, adjlist, sparse_matrix) = load_train_data("data/Celegans.txt", delimiter=' ')

    print("Loading test data...")
    (test, test_dict) = load_test_data("data/Celegans_test.txt", delimiter=' ', with_index=False)

    print("Sampling negative links...")
    neg_links = sample_negative_links(n=len(adjlist.keys()), size=len(train), adjlist=adjlist)

    feature_vector_path = "train_feature_vectors.txt"

    #print("Saving positive train data...")
    #save_train_feature_vectors(feature_vector_path, train_data=train, label=1, network=sparse_matrix)

    #print("Saving negative train data...")
    #save_train_feature_vectors(feature_vector_path, train_data=neg_links, label=0, network=sparse_matrix)

    print("Training...")
    model = train_svm(feature_vector_path, network=sparse_matrix)

    print("Testing...")
    test_svm(model, test_data=test, test_dict=test_dict, network=sparse_matrix)

if __name__ == "__main__":
    main()
                 