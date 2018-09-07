ng.py, includes feature extraction:

python ng.py -t extract_pos
python ng.py -t extract_neg

data will be stored in /data

python ng.py -t train
will train and save a tensorflow model
python ng.py -t predict
will load the model and predict results

python ng.py -t edgelist
will convert train data into edgelist

python embedding_tensor.py
will load up emb files and train the network