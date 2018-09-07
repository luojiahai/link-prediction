import numpy as np
import math
import networkx as nx
import os.path
import time
import pickle
from tqdm import tqdm
import sys
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

def load_predict_data(path, delimiter):
    bar = tqdm()
    test = []
    # test_dict = {}
    with open(path, "r") as f:
        for line in f:
            # split the line
            splited = line.rstrip().split(delimiter)
            v0 = int(splited[0])
            v1 = int(splited[1])
            v2 = int(splited[2])
            test.append((v0,v1, v2))
            # progress bar update
            bar.update(1)
        print('\n')
    bar.close()
    return test

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
def load_exist_extracted(path, adjlist):
    feature_list = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                numline = int(line.strip())
                # del adjlist[numline]
    return feature_list
# ja,re,cn,aai,size_src,size_sink
def feature_extraction(network,link,expend=False):
    feature = []
    (x, y) = link
    common_neighbors = len(list(nx.common_neighbors(network, x, y)))
    (_, _, adamic_adar_index) = list(nx.adamic_adar_index(network, [(x, y)]))[0]
    if(expend == False):
        # jaccard coefficient
        (_, _, jaccard_coefficient) = list(nx.jaccard_coefficient(network, [(x, y)]))[0]
        # resource allocation index
        (_, _, resource_allocation_index) = list(nx.resource_allocation_index(network, [(x, y)]))[0]
        return jaccard_coefficient,resource_allocation_index,common_neighbors,adamic_adar_index,len(adjlist[x]),len(adjlist[y])
    else:
        return common_neighbors,adamic_adar_index,len(adjlist[x]),len(adjlist[y])
def process_pos(network, base_node, connected_node):
    feature_set = []
    # print(f"-----------------------------------------------\nbase: {base_node}, edges:{connceted_node}")

    for e in tqdm(connected_node):
        ja,re,cn,aai,size_src,size_sink = feature_extraction(network,(base_node,e))
        feature_set.append((e,ja,re,cn,aai,size_src,size_sink))

    # mymodule.q.put(feature_set)
    return feature_set
def reshape_adj(adjlist, extracted_node_list,size=-1, max_adjlist_size=2000):
    adj = []
    counter = 0
    adj_collision_holder = {}
    # temperarily cutout already extracted elements in extracted nodes
    for e in extracted_node_list:
        adj_collision_holder[e] = adjlist[e]
        del adjlist[e]
    adjlist_sample = adjlist if size < 0 else random.sample(list(adjlist),size)
    for e in adjlist_sample:
        base,connected = e
        if(len(connected)>max_adjlist_size):
            connected = connected[:int(max_adjlist_size/len(connected)*max_adjlist_size) + 1]
        adj.append((base,connected))
        counter = counter + 1
        if(counter == size):
            break
    # push elements back into adjlist 
    for e in adj_collision_holder:
        adjlist[e] = adj_collision_holder[e]
    adj_collision_holder = None
    return adj

# shape adj list into blocks list
def reshap_adj_to_block(adjlist, blocksize):
    block_buffer = []
    counter = 0
    block_adj = []
    for e in adjlist:
        block_buffer.append(e)
        counter = counter + 1
        if(counter == blocksize):
            counter = 0
            block_adj.append(block_buffer)
            block_buffer = []
    return block_adj

def file_lines_count(path):
    counter = 0
    with open(path,'r') as f:
        for line in f:
            counter = counter + 1

    return counter
# def load_extracted_pos(path):
#     extracted_pos = []
#     if(os.path.exists(path)):
#         with open(path,'r') as f:
#             for line in f:
#                 s = line.rstrip().split(" ")
#                 extracted_pos.append((int(s[0]),int(s[1]),s[2],s[3]))
#     return extracted_pos
def load_extracted_pos_v2(path):
    extracted_pos = []
    if(os.path.exists(path)):
        with open(path,'r') as f:
            for line in f:
                s = line.rstrip().split(" ")
                extracted_pos.append((int(s[0]),int(s[1]),float(s[2]),float(s[3]),float(s[4]),float(s[5]),int(s[6]),int(s[7])))
    return extracted_pos
# def load_extracted_neg(path):
#     extracted_ng = {}
#     if(os.path.exists(path)):
            
#         with open(path,'r') as f:
#             for line in f:
#                 s = line.rstrip().split(' ')
#                 extracted_ng[s[0]]=(float(s[1]),float(s[2]))
#     return extracted_ng
# extracted neg - formet bas_enode|disconnected_node ja re
def load_extracted_neg_v2(path):
    extracted_ng = {}
    if(os.path.exists(path)):
            
        with open(path,'r') as f:
            for line in f:
                s = line.rstrip().split(' ')
                extracted_ng[s[0]]=(float(s[1]),float(s[2]),float(s[3]),float(s[4]),int(s[5]),int(s[6]))
    return extracted_ng
# def initProcess(q):
# #   mymodule.network = share
#   mymodule.q = q

def processe_data_tensor_train(pos,neg):
    data = []
    labels = []
    for x in tqdm(range(len(pos))):
        # print(f"pos:{pos[x]}, neg:{neg[x]}")
        _,_,ja_p,re_p,cn_p,aai_p,size_src_p,size_sink_p = pos[x]
        ja_n,re_n,cn_n,aai_n,size_src_n,size_sink_n = neg[x]

        data.append(np.array([ja_p,re_p]))
        labels.append(True)
        data.append(np.array([ja_n,re_n]))
        labels.append(False)
    return np.array(data),np.array(labels)

# def block_pos_processing(adj_block):
#     for base_node,connected_node in adj_block:
#         for e,ja,re in process_pos(network,base_node, connected_node,False):
#             q.put((base_node,e,ja,re))
# def block_writing():
#     base_node_record = {}
#     with open(extracted_node_path,'a') as extracted_node_file:
#         with open(feature_set_path, 'a') as feature_set_file:
#             for base_node,connected,ja,re  in iter(q.get, None):
#                 # print(base_node)
#                 feature_set_file.write(f"{base_node} {connected} {ja} {re}\n")
#                 if(base_node not in base_node_record):
#                     base_node_record[base_node] = True
#                     extracted_node_file.write(f"{base_node}\n")
#                     extracted_node_file.flush()
#                 feature_set_file.flush()

def convert_lou_predict(path):
    index = 1
    with open(extracted_predict_path,'w') as o:
        with open(path,'r') as f:
            for line in f:
                src,sink,ja,re = line.rstrip().split("\t")
                o.write(f"{index} {src} {sink} {ja} {re}\n")
                index = index + 1
def sample_neg(to_process, extract_neg):
    tqdmbar = tqdm(total=to_process)
    print("Sampling negative set")
    neg_set = []
    counter = 0
    adj_to_list = list(adjlist)
    # construct negative list
    while counter < to_process:
        x = random.randrange(len(adj_to_list))
        y = random.randrange(len(adj_to_list))
        if (x!=y and f"{adj_to_list[x]}|{adj_to_list[y]}" not in extract_neg):
            neg_set.append((adj_to_list[x],adj_to_list[y]))
            counter = counter + 1
            tqdmbar.update(1)
    tqdmbar.close()
    return neg_set
def main(task):
    if(task == 'train'):
        print("Loading negative instances")
        extracted_neg = list(load_extracted_neg_v2(extracted_neg_v2_path).values())
        print("Loading positive instances")
        extracted_pos = load_extracted_pos_v2(extracted_pos_v2_path)
        print("Loading predict data...")
        # processing data
        print("Processing data")
        data,labels = processe_data_tensor_train(extracted_pos,extracted_neg)
        # Start training
        # categorical_labels = to_categorical(labels, num_classes=2)
        print("Building model")
        model = keras.Sequential()
        # Adds a densely-connected layer with 64 units to the model:
        model.add(keras.layers.Dense(64, activation='relu', input_dim=2))
        # Add another:
        model.add(keras.layers.Dense(64, activation='relu'))
        # Add a softmax layer with 10 output units:
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

        model.fit(data, labels, epochs=500, batch_size=50000)
        
        # save to disk, from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        scores = model.evaluate(data, labels, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as j:
            j.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Model saved")
    elif(task == 'extract_pos'):

            print("Loading already extracted node features")
            existing_feature_list = load_exist_extracted(extracted_node_path,adjlist)
            
            print("Continuing feature extraction")
            # Multi threading, not quite working....
            # disk_writer = Process(target=block_writing)
            # disk_writer.daemon = True
            # disk_writer.start()
            # pool = Pool(processes=4)
            # # print("Processing positive datas")
            # # partial_pos = partial(process_pos, network)
            print("Sampling nodes")
            reshaped = reshape_adj(adjlist.items(),existing_feature_list,size=300000)
            # reshaped = reshap_adj_to_block(reshaped,1)
            # for _ in tqdm(pool.imap(block_pos_processing, reshaped), total=len(reshaped)):
            #     pass
            # print("process complete.")

            with open(extracted_node_path,'a') as extracted_node_file:
                with open(extracted_pos_v2_path, 'a') as feature_set_file:
                    for base_node,connected_node in tqdm(reshaped):
                        # extract feature
                        for e,ja,re,cn,aai,size_src,size_sink  in process_pos(network,base_node, connected_node):
                            feature_set_file.write(f"{base_node} {e} {ja} {re} {cn} {aai} {size_src} {size_sink}\n")
                        feature_set_file.flush()
                        extracted_node_file.write(f"{base_node}\n")
                        extracted_node_file.flush()
    elif(task == 'extract_neg'):
        # count extracted
        count = file_lines_count(extracted_pos_v2_path)
        extracted_neg = load_extracted_neg_v2(extracted_neg_v2_path)
        # do nothing if have already enough negs
        if(len(extracted_neg) < count):
            to_process = count - len(extracted_neg)
            

            # extract features and save to disk
            with open(extracted_neg_v2_path,'a') as f:
                for n1,n2 in tqdm(sample_neg(to_process,extracted_neg)):
                    # print(f"neg_counter{neg_counter} < to_process{to_process}:{(neg_counter < to_process)}")
                    ja,re,cn,aai,size_src,size_sink = feature_extraction(network,(n1,n2))
                    f.write(f"{n1}|{n2} {ja} {re} {cn} {aai} {size_src} {size_sink}\n")
    elif(task == 'extract_predict'):
        predict = load_predict_data("data/twitter_test.txt", delimiter='\t')
        with open(extracted_predict_path,'w') as f:
            for index,src,sink in tqdm(predict):
                ja,re,cn,aai,size_src,size_sink = feature_extraction(network,(src,sink))
                f.write(f"{index} {src} {sink} {ja} {re} {cn} {aai} {size_src} {size_sink}\n")
    elif(task == 'edgelist'):
        nx.write_edgelist(network,extracted_edgelist_path)
        # neg_set = sample_neg(len(network.edges()),[])
        # with open(extracted_edgelist_path) as edge_file:
        #     with open(extracted_edgelist_label_path) as label_file:
        #         for (pos_src,pos_sink),(neg_src,neg_sink) in zip(network.edges(),neg_set):
        #             edge_file.write(f"{pos_src} {pos_sink}\n")

    elif(task == 'predict'):
        predict_data = []
        with open(extracted_predict_path, 'r') as f:
            for line in tqdm(f):
                index,_,_,ja,re,cn,aai,size_src,size_sink= line.rstrip().split(' ')
                predict_data.append(np.array([re]))
        predict_data = np.array(predict_data)
        # load json and create model

        # save to disk, from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
        # score = loaded_model.evaluate(X, Y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        # convert_lou_predict("predict_feature_vectors.txt")
        predicts = loaded_model.predict(predict_data)
        print("Predict complete, saving to disk")
        with open(predict_output_path, 'w') as o:
            o.write("Id,Prediction\n")
            for i,e in enumerate(predicts):
                o.write(f"{i+1},{e[0]}\n")
    # elif(task == 'expand'):
    #     print("Expending positive feature sets")
    #     with open(extracted_pos_v2_path,'w') as o:
    #         for src,sink,ja,re in tqdm(load_extracted_pos(feature_set_path)):
    #             # print(f"{src} {sink} {ja} {re}")
    #             cn,aai,size_src,size_sink = feature_extraction(network,(src,sink),expend=True)
    #             o.write(f"{src} {sink} {ja} {re} {cn} {aai} {size_src} {size_sink}\n")
    #     print("Expending negative feature sets")

    #     with open(extracted_neg_v2_path,'w') as o:
    #         for src_sink,ja,re in tqdm(load_extracted_neg(extracted_neg_path)):
    #             src,sink=src_sink.split("|")
    #             cn,aai,size_src,size_sink = feature_extraction(network,(src,sink),expend=True)
    #             o.write(f"{src} {sink} {ja} {re} {cn} {aai} {size_src} {size_sink}\n")


        
if __name__ == "__main__":
    persistent_network_path = "./data/persistent_network.pst"
    extracted_node_path = "./data/extracted.ext"
    # feature_set_path = "./data/featureset.ext"
    extracted_pos_v2_path = "./data/extracted_pos_v2.ext"
    # extracted_neg_path = "./data/extracted_neg.ext"
    extracted_neg_v2_path = "./data/extracted_neg_v2.ext"
    extracted_predict_path = "./data/extracted_predict.ext"
    extracted_edgelist_path = "./data/edgelist.ext"
    extracted_edgelist_label_path = "./data/edgelist_label.ext"
    predict_output_path = "./data/predict_output.ext"
    for i in range(len(sys.argv)):
        if (sys.argv[i] == '-t'):
            task = sys.argv[i+1]
    
    if(task != "train" and task != "predict"):
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
    # manager = Manager()
    # q = Queue()
    main(task)
