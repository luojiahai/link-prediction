import time
import progressbar
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
from random import shuffle


def get_edge_embeddings(edge_list, emb_matrix, flag=False):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        edge_emb = []
        if (flag):
            if ((node1 not in emb_matrix) or (node2 not in emb_matrix)):
                edge_emb = np.zeros(128)
            else:
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
        elif ((node1 not in emb_matrix) or (node2 not in emb_matrix)):
            continue
        else:
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def filter_edge(edgelist, embs, size):
    filtered = []
    for edge in edgelist:
        x = edge[0]
        y = edge[1]
        if ((x in embs) and (y in embs)):
            filtered.append(edge)
    shuffle(filtered)
    return filtered[:size]

def load_predict(path):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    train = []
    with open(path, "r") as f:
        count = 0
        for line in f:
            splited = splited = line.rstrip().split('\t')
            x = int(splited[1])
            y = int(splited[2])
            train.append([x, y])
            count += 1
            bar.update(count)
    return train

def load_train(path):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    train = []
    with open(path, "r") as f:
        count = 0
        for line in f:
            splited = splited = line.rstrip().split('\t')
            x = int(splited[0])
            y = int(splited[1])
            train.append([x, y])
            count += 1
            bar.update(count)
    return train

def load_embs(path):
    embs = {}
    with open(path, "r") as f:
        count = 0
        first_line = f.readline()
        first_line_splited = first_line.rstrip().split(' ')
        n_nodes = int(first_line_splited[0])
        bar = progressbar.ProgressBar(max_value=n_nodes)
        for line in f:
            splited = splited = line.rstrip().split(' ')
            vector = []
            for i in range(1, len(splited)):
                vector.append(float(splited[i]))
            embs[int(splited[0])] = vector
            count += 1
            if (count == n_nodes):
                time.sleep(0.1)
            bar.update(count)
    return embs

def main():
    print("Loading embeddings...")
    embs = load_embs(path="data/train.emb")
    print("\n")

    print("Loading positive train data...")
    train_pos = load_train(path="data/train_pos.txt")
    print("\n")

    print("Loading negative train data...")
    train_neg = load_train(path="data/train_neg.txt")
    print("\n")

    print("Loading test data...")
    test = load_predict(path="data/twitter_test.txt")
    print("\n")

    print("Filtering data...")
    train_pos_filtered = filter_edge(train_pos, embs, size=100000)
    print("Train positive data original size: " + str(len(train_pos)))
    print("Train positive data filtered size: " + str(len(train_pos_filtered)))
    train_neg_filtered = filter_edge(train_neg, embs, size=100000)
    print("Train negative data original size: " + str(len(train_neg)))
    print("Train negative data filtered size: " + str(len(train_neg_filtered)))
    print("\n")

    print("Getting edge embeddings...")
    # Train-set edge embeddings
    train_pos_embs = get_edge_embeddings(train_pos_filtered, embs)
    train_neg_embs = get_edge_embeddings(train_neg_filtered, embs)
    train_edge_embs = np.concatenate([train_pos_embs, train_neg_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_pos_filtered)), np.zeros(len(train_neg_filtered))])

    # Test-set
    test_edge_embs = get_edge_embeddings(test, embs, flag=True)
    print("\n")

    print("Training...")
    # Train
    classifier = LogisticRegression(random_state=0)
    #classifier = MLPClassifier()
    classifier.fit(train_edge_embs, train_edge_labels)
    print("\n")

    print("Evaluating...")
    of = open("predict.csv", "w")
    of.write("Id,Prediction" + '\n')
    i = 1
    for edge_emb in test_edge_embs:
        results = classifier.predict_proba([edge_emb])[0]
        prob_per_class_dictionary = dict(zip(classifier.classes_, results))
        string = str(i) + ',' + str(prob_per_class_dictionary[1])
        of.write(string + '\n')
        i += 1
    print("\n")

    return None

if __name__ == "__main__":
    main()