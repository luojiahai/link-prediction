import math
import networkx as nx
from scipy import sparse
import numpy as np
from node2vec import Node2Vec

def main():
    # generate directed graph from training dataset
    graph = nx.read_adjlist("train.txt", create_using=nx.DiGraph())
    node2vec = Node2Vec(graph, dimensions)
    return None

if __name__ == "__main__":
    main()