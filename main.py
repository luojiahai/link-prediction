import math
import networkx as nx
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import random
import time
import progressbar
from random import shuffle


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def save_train_feature_vectors(path, train_pos, train_neg, network, size=None):
    f = open(path, "w")
    i = 0
    end = len(train_pos)+len(train_neg)
    if (size):
        end = size
    print("Saving positive links...")
    bar_pos = progressbar.ProgressBar(max_value=int(end / 2))
    while i < (end / 2):
        rand = random.randrange(len(train_pos))
        randed = []
        while rand in randed:
            rand = random.randrange(len(train_pos))
        randed.append(rand)
        (x, y) = train_pos[rand]
        feature = feature_extraction((x, y), network)
        string = str(x) + '\t' + str(y) + '\t' + str(1)
        for elem in feature:
            string += '\t' + str(elem)
        f.write(string + '\n')
        i = i + 1
        if (i == int(end / 2)):
            time.sleep(0.1)
        bar_pos.update(i)
    print('\n')
    print("Saving negative links...")
    bar_neg = progressbar.ProgressBar(max_value=int(end / 2))
    while i < end:
        rand = random.randrange(len(train_neg))
        randed = []
        while rand in randed:
            rand = random.randrange(len(train_neg))
        randed.append(rand)
        (x, y) = train_neg[rand]
        feature = feature_extraction((x, y), network)
        string = str(x) + '\t' + str(y) + '\t' + str(0)
        for elem in feature:
            string += '\t' + str(elem)
        f.write(string + '\n')
        i = i + 1
        if (i == end):
            time.sleep(0.1)
        bar_neg.update(i - int(end / 2))
    print('\n')
    return None

def save_test_feature_vectors(path, test_data, network):
    bar = progressbar.ProgressBar(max_value=len(test_data))
    i = 0
    f = open(path, "w")
    for ((x, y), label) in test_data.items():
        feature = feature_extraction((x, y), network)
        string = str(x) + '\t' + str(y) + '\t' + str(label)
        for elem in feature:
            string += '\t' + str(elem)
        f.write(string + '\n')
        # progress bar update
        i = i + 1
        if (i == len(test_data)):
            time.sleep(0.1)
        bar.update(i)
    print('\n')
    return None

def save_predict_feature_vectors(path, predict_data, network):
    bar = progressbar.ProgressBar(max_value=len(predict_data))
    i = 0
    f = open(path, "w")
    for (x, y) in predict_data:
        feature = feature_extraction((x, y), network)
        string = str(x) + '\t' + str(y)
        for elem in feature:
            string += '\t' + str(elem)
        f.write(string + '\n')
        # progress bar update
        i = i + 1
        if (i == len(predict_data)):
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
            label = int(splited[2])
            feature = []
            for i in range(3, len(splited)):
                feature.append(float(splited[i]))
            X.append(feature)
            y.append(label)
    model = None
    if (model_name == "svm_rbf"):
        model = SVC(C=8.0, kernel='rbf', probability=True)
    elif (model_name == "svm_linear"):
        model = SVC(C=8.0, kernel='linear', probability=True)
    elif (model_name == "dt"):
        model = DecisionTreeClassifier()
    elif (model_name == "lg"):
        model = LogisticRegression()
    elif (model_name == "knn"):
        model = KNeighborsClassifier(n_neighbors=12)
    elif (model_name == "bagging"):
        model = BaggingClassifier()
    elif (model_name == "mlp"):
        model = MLPClassifier()
    model.fit(X, y)
    return model

def test_sklearn(fitted_model, test_data, network):
    bar = progressbar.ProgressBar(max_value=len(test_data))
    i = 1
    y_true = []
    y_score = []
    f = open("test_feature_vectors.txt", "r")
    for line in f:
        splited = line.rstrip().split('\t')
        label = int(splited[2])
        feature = []
        for ind in range(3, len(splited)):
            feature.append(float(splited[ind]))
        results = fitted_model.predict_proba([feature])[0]
        prob_per_class_dictionary = dict(zip(fitted_model.classes_, results))
        y_true.append(label)
        y_score.append(prob_per_class_dictionary[1])
        # progress bar update
        i = i + 1
        if (i == len(test_data)+1):
            time.sleep(0.1)
        bar.update(i - 1)

    print('\n')
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(y_score), pos_label=1)
    print("AUC: " + str(metrics.auc(fpr, tpr)))
    return None

def predict_sklearn(fitted_model, predict_data, network, path):
    of = open(path, "w")
    f = open("predict_feature_vectors.txt", "r")
    bar = progressbar.ProgressBar(max_value=len(predict_data))
    of.write("Id,Prediction\n")
    i = 1
    for line in f:
        splited = line.rstrip().split('\t')
        feature = []
        for ind in range(2, len(splited)):
            feature.append(float(splited[ind]))
        results = fitted_model.predict_proba([feature])[0]
        prob_per_class_dictionary = dict(zip(fitted_model.classes_, results))
        string = str(i) + ',' + str(prob_per_class_dictionary[1])
        of.write(string + '\n')
        # progress bar update
        i = i + 1
        if (i == len(predict_data)+1):
            time.sleep(0.1)
        bar.update(i - 1)
    print('\n')
    return None

def feature_extraction(link, network):
    feature = []
    (x, y) = link
    
    # common neighbors
    #common_neighbors = len(list(nx.common_neighbors(network, x, y)))
    #feature.append(sigmoid(common_neighbors))
    # jaccard coefficient
    (_, _, jaccard_coefficient) = list(nx.jaccard_coefficient(network, [(x, y)]))[0]
    feature.append(sigmoid(jaccard_coefficient))
    # preferential attachment
    #(_, _, preferential_attachment) = list(nx.preferential_attachment(network, [(x, y)]))[0]
    #feature.append(sigmoid(preferential_attachment))
    # adamic adar index
    #(_, _, adamic_adar_index) = list(nx.adamic_adar_index(network, [(x, y)]))[0]
    #feature.append(sigmoid(adamic_adar_index))
    # resource allocation index
    (_, _, resource_allocation_index) = list(nx.resource_allocation_index(network, [(x, y)]))[0]
    feature.append(sigmoid(resource_allocation_index))
    
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

def load_train_data(test_size, path, delimiter):
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
    (train_pos, train_neg, test, network) = load_train_data(test_size=2000, path="data/Celegans.txt", delimiter=' ')

    print("Loading predict data...")
    (predict, predict_dict) = load_predict_data("data/Celegans_test.txt", delimiter=' ', with_index=False)

    flag = True
    if (flag):
        print("Saving train data...")
        save_train_feature_vectors(path="train_feature_vectors.txt", train_pos=train_pos, train_neg=train_neg, network=network, size=400)

        print("Saving test data...")
        save_test_feature_vectors(path="test_feature_vectors.txt", test_data=test, network=network)

        print("Saving predict data...")
        save_predict_feature_vectors(path="predict_feature_vectors.txt", predict_data=predict, network=network)

    models = ["svm_rbf", "svm_linear", "knn", "bagging", "mlp", "lg"]
    for model_name in models:
        print("Training " + model_name + "...")
        model = train_sklearn(model_name, path="train_feature_vectors.txt")

        test_flag = True
        if (test_flag):
            print("Testing " + model_name + "...")
            test_sklearn(model, test_data=test, network=network)

        output_file_name = "predict_output_" + model_name + ".csv" 
        print("Predicting " + model_name + "...")
        predict_sklearn(model, predict_data=predict, network=network, path=output_file_name)

if __name__ == "__main__":
    main()
                 