import networkx as nx

def main():
    # generate directed graph from training dataset
    graph = nx.read_adjlist("data/Celegans_adj.txt", create_using=nx.DiGraph())
    

    return None

if __name__ == "__main__":
    main()