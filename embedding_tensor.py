import time
from sklearn.linear_model import LogisticRegression
import numpy as np
from random import shuffle
import math
import networkx as nx
import os.path
import time
import pickle
from multiprocessing import Pool, Array, Process, Queue, Manager
from tqdm import tqdm
from functools import partial 
import sys
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
import copy



def get_edge_embeddings(edge_list, emb_matrix):
    embs = []
    for edge in tqdm(edge_list):
        try:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
        except:
            edge_emb = np.zeros(128)
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
    bar = tqdm()
    train = []
    with open(path, "r") as f:
        count = 0
        for line in f:
            splited = splited = line.rstrip().split('\t')
            x = int(splited[1])
            y = int(splited[2])
            train.append([x, y])
            count += 1
            bar.update(1)
    bar.close()
    return train


def load_embs(path):
    embs = {}
    with open(path, "r") as f:
        first_line = f.readline()
        first_line_splited = first_line.rstrip().split(' ')
        n_nodes = int(first_line_splited[0])
        bar = tqdm() 
        for line in f:
            splited = splited = line.rstrip().split(' ')
            vector = []
            for i in range(1, len(splited)):
                vector.append(float(splited[i]))
            embs[int(splited[0])] = vector
            bar.update(1)
    return embs
def load_train_data(path, delimiter):
    bar = tqdm()
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
            bar.update(1)
        print('\n')
    bar.close()
    return adjlist, network
def sample_neg(to_process, adjlist):
    tqdmbar = tqdm(total=to_process)
    print("Sampling negative set")
    neg_set = {} 
    counter = 0
    adj_to_list = list(adjlist)
    # construct negative list
    while counter < to_process:
        x = random.randrange(len(adj_to_list))
        y = random.randrange(len(adj_to_list))
        key = f"{adj_to_list[x]}|{adj_to_list[y]}"
        if (x!=y and key not in neg_set):
            neg_set[key ] = ((adj_to_list[x],adj_to_list[y]))
            counter = counter + 1
            tqdmbar.update(1)
    tqdmbar.close()
    return neg_set
def main():
    print("Loading embeddings...")
    embs = load_embs(path="data/train.emb")
    print("\n")
    persistent_network_path = "./data/persistent_network_embedding.pst"
    predict_output_path = "./data/predict_output.ext"
    if os.path.exists(persistent_network_path):
        print("Persistant traning dataset found, loading")
        start_time = time.time()
        with open(persistent_network_path, 'rb') as f:
            adjlist,network = pickle.load(f)
        elapsed_time = time.time() - start_time
        print(f"Loaded in {elapsed_time}s")
    else:
        # falling back to normal dataset
        print("Persistant traning network not found, loading from dataset")
        adjlist,network = load_train_data(path="data/train_filtered.txt", delimiter='\t')
        
        print("Complete. Dumping object to disk...")
        with open(persistent_network_path, 'wb') as f:
            pickle.dump((adjlist,network), f, pickle.HIGHEST_PROTOCOL)


    print("Loading test data...")
    test = load_predict(path="data/twitter_test.txt")
    # print("\n")
    sample_size = 1000000
    print("Sampling positive")
    edgelist = list(network.edges())

    pos = random.sample(edgelist,sample_size)
    neg = sample_neg(sample_size,edgelist)
    

    print("Getting edge embeddings...")
    # Train-set edge embeddings
    train_pos_embs = get_edge_embeddings(pos, embs)
    train_neg_embs = get_edge_embeddings(neg.values(), embs)
    pos = None
    neg = None
    training_feature = []
    training_label = []
    print("Final data processing")
    for i in tqdm(range(sample_size)):
        training_feature.append(np.array(train_pos_embs[i]))
        training_label.append(True)
        training_feature.append(np.array(train_neg_embs[i]))
        training_label.append(False)
    train_neg_embs = None
    train_neg_embs = None
    # # Test-set
    # test_edge_embs = get_edge_embeddings(test, embs)
    # print("\n")


    # Train
    print("Building model")
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(32, activation='relu', input_dim=1))
    print("First layer complete")
    # Add another:
    # model.add(keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print("Compiling")
    model.compile(optimizer='nadam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    print("Training...")
    model.fit(training_feature, training_label, epochs=500, batch_size=50000)

    scores = model.evaluate(training_feature, training_label, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as j:
        j.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Model saved")    
    return None

if __name__ == "__main__":
    main()