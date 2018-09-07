Requirement:
numpy
networkx
pickle
tqdm
random
tensorflow


path:
train.txt in /data/train.txt
public-test.txt in /data/twitter_test.txt


ng.py, includes feature extraction:

python ng.py -t extract_pos
python ng.py -t extract_neg
python ng.py -t extract predict

data will be stored in /data

Then, train the model

python ng.py -t train
will train and save a tensorflow model
python ng.py -t predict
will load the model and predict results

python ng.py -t edgelist
will convert train data into edgelist

python embedding_tensor.py 
will load up emb files and train the network

emb file should be generated from node2vec or deepwalk