import math
import networkx as nx
from scipy import sparse
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
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
    
def train_sklearn(model_name, path):
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = None
    if (model_name == "svm"):
        model = SVC(kernel='rbf', probability=True)
    elif (model_name == "dt"):
        model = tree.DecisionTreeClassifier()
    model.fit(X, y)
    return model

def test_sklearn(fitted_model, test_data, network):
    bar = progressbar.ProgressBar(max_value=len(test_data)+1)
    i = 1
    y_true = []
    y_pred = []
    for (link, label) in test_data.items():
        feature = feature_extraction(link, network)
        results = fitted_model.predict_proba([feature])[0]
        prob_per_class_dictionary = dict(zip(fitted_model.classes_, results))
        y_true.append(label)
        y_pred.append(prob_per_class_dictionary[1])
        # progress bar update
        i = i + 1
        if (i == len(test_data)+1):
            time.sleep(0.1)
        bar.update(i)
    print('\n')
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(y_pred), pos_label=1)
    print("AUC: " + str(metrics.auc(fpr, tpr)))
    return None

def predict_sklearn(fitted_model, predict_data, network, path):
    with open(path, "w") as f:
        bar = progressbar.ProgressBar(max_value=len(predict_data)+1)
        f.write("Id,Prediction\n")
        i = 1
        for (x, y) in predict_data:
            feature = feature_extraction((x, y), network)
            results = fitted_model.predict_proba([feature])[0]
            prob_per_class_dictionary = dict(zip(fitted_model.classes_, results))
            string = str(i) + ',' + str(prob_per_class_dictionary[1])
            f.write(string + '\n')
            # progress bar update
            i = i + 1
            if (i == len(predict_data)+1):
                time.sleep(0.1)
            bar.update(i)
        print('\n')
    return None

def feature_extraction(link, network):
    feature = []
    (x, y) = link
    
    # common neighbors
    #common_neighbors = len(list(nx.common_neighbors(network, x, y)))
    #feature.append(common_neighbors)
    # jaccard coefficient
    (_, _, jaccard_coefficient) = list(nx.jaccard_coefficient(network, [(x, y)]))[0]
    feature.append(jaccard_coefficient)
    ## preferential attachment
    #(_, _, preferential_attachment) = list(nx.preferential_attachment(network, [(x, y)]))[0]
    #feature.append(preferential_attachment)
    # adamic adar index
    #(_, _, adamic_adar_index) = list(nx.adamic_adar_index(network, [(x, y)]))[0]
    #feature.append(adamic_adar_index)
    # resource allocation index
    (_, _, resource_allocation_index) = list(nx.resource_allocation_index(network, [(x, y)]))[0]
    feature.append(resource_allocation_index)
    
    return feature

def sample_negative_links(size, n, adjlist):
    neg_links = []
    i = 0
    while i < size:
        x = random.randrange(n)
        y = random.randrange(n)
        if (y not in adjlist[x]):
            neg_links.append((x, y))
            i = i + 1
    return neg_links

def load_predict_data(path, delimiter, with_index):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    pb_i = 0
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
            # progress bar update
            pb_i = pb_i + 1
            bar.update(pb_i)
        print('\n')
    return (test, test_dict)

def load_train_data(test_ratio, path, delimiter):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    pb_i = 0

    train_pos = []
    test = {}
    adjlist = {}    # adjacency list
    network = nx.Graph()
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
                # construct network
                network.add_edge(v1, v2)
                # construct (x, y) tuple
                train_pos.append((v1, v2))
            # progress bar update
            pb_i = pb_i + 1
            bar.update(pb_i)
        print('\n')

    print("Sampling negative links...")
    train_neg = sample_negative_links(len(train_pos), len(adjlist.keys()), adjlist)

    test_size = (len(train_pos) + len(train_neg)) * test_ratio
    i_pos = 0
    while i_pos < test_size/2:
        rand = random.randrange(len(train_pos))
        if train_pos[rand] not in test:
            test[train_pos[rand]] = 1
            i_pos = i_pos + 1
    i_neg = 0
    while i_neg < test_size/2:
        rand = random.randrange(len(train_neg))
        if train_neg[rand] not in test:
            test[train_neg[rand]] = 0
            i_neg = i_neg + 1

    return (train_pos, train_neg, test, network)

def main():
    print("Loading train and test data...")
    (train_pos, train_neg, test, network) = load_train_data(test_ratio=0.1, path="data/Celegans.txt", delimiter=' ')

    print("Loading predict data...")
    (predict, predict_dict) = load_predict_data("data/Celegans_test.txt", delimiter=' ', with_index=False)

    feature_vector_path = "feature_vectors.txt"

    print("Saving positive train data...")
    save_train_feature_vectors(feature_vector_path, train_data=train_pos, label=1, network=network)

    print("Saving negative train data...")
    save_train_feature_vectors(feature_vector_path, train_data=train_neg, label=0, network=network)

    print("Training...")
    model = train_sklearn("svm", feature_vector_path)

    print("Testing...")
    test_sklearn(model, test_data=test, network=network)

    print("Predicting...")
    predict_sklearn(model, predict_data=predict, network=network, path="predict_output_svm_rbf.txt")

if __name__ == "__main__":
    main()
                 