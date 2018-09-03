import numpy as np
import math
import networkx as nx
import progressbar
import os.path
import time
import pickle
from multiprocessing import Pool, Array, Process, Queue
from tqdm import tqdm
from functools import partial 
import myModule as mymodule

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

def load_train_data(path, delimiter):
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

    return adjlist, network
def load_exist_extracted(path, adjlist):
    feature_list = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                numline = int(line.strip())
                del adjlist[numline]
    return feature_list

def feature_extraction(network,link):
    feature = []
    (x, y) = link

    # jaccard coefficient
    (_, _, jaccard_coefficient) = list(nx.jaccard_coefficient(network, [(x, y)]))[0]

    # resource allocation index
    (_, _, resource_allocation_index) = list(nx.resource_allocation_index(network, [(x, y)]))[0]
    
    return jaccard_coefficient,resource_allocation_index

def process_pos(network, base_node, connected_node):
    feature_set = []
    # print(f"-----------------------------------------------\nbase: {base_node}, edges:{connceted_node}")
    for e in tqdm(connected_node):
        try:
            ja,re = feature_extraction(network,(base_node,e))
            feature_set.append((e,ja,re))
            # print(f"For pair {base_node}:{e}, j:{ja}, re:{re}")
        except:
            # print(f"Error occured")
            pass
    # mymodule.q.put(feature_set)
    return feature_set
# shape adj list into blocks list
def reshape_adj(adjlist, size):
    adj = []
    counter = 0
    for e in adjlist:
        adj.append(e)
        counter = counter + 1
        if(counter == size):
            break


    return adj

# def initProcess(q):
# #   mymodule.network = share
#   mymodule.q = q

def main():

    persistent_network_path = "./data/persistent_network.pst"
    extracted_node_path = "./data/extracted.ext"
    feature_set_path = "./data/featureset.ext"
    # attempt to load from persistant dataset
    print("Loading train data...")
    
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
        adjlist,network = load_train_data(path="data/train.txt", delimiter='\t')
        print("Complete. Dumping object to disk...")
        with open(persistent_network_path, 'wb') as f:
            pickle.dump((adjlist,network), f, pickle.HIGHEST_PROTOCOL)

    print("Loading predict data...")
    # (predict, predict_dict) = load_predict_data("data/twitter_test.txt", delimiter='\t', with_index=True)
    
    print("Loading already extracted node features")
    existing_feature_list = load_exist_extracted(extracted_node_path,adjlist)
    
    print("Continuing feature extraction")
    # print("Reshaping input structure")
    # reshaped_adj = reshape_adj(adjlist.items(),4)[:40]
    # q = Queue()
    # pool = Pool(processes=2,initializer=initProcess,initargs=(q,))
    # print("Processing positive datas")
    # partial_pos = partial(process_pos, network)

    # for _ in tqdm.tqdm(pool.imap(partial_pos, reshaped_adj), total=len(reshaped_adj)):
    #     pass

    reshaped = reshape_adj(adjlist.items(),10)
    with open(extracted_node_path,'a') as extracted_node_file:
        with open(feature_set_path, 'a') as feature_set_file:
            for base_node,connected_node in tqdm(reshaped):
                # extract feature
                extracted_node_file.write(f"{base_node}\n")
                for e,ja,re in process_pos(network,base_node, connected_node):
                    feature_set_file.write(f"{base_node} {e} {ja} {re}\n")




if __name__ == "__main__":
    main()