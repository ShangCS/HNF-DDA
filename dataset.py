import torch
import pandas as pd
import networkx as nx

class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None
        self.adj = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_dataset(data_dir, dataname):
    if dataname in ('HetioNet', 'KEGG'):
        dataset = load_HN_dataset(data_dir, dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_HN_dataset(data_dir, name):
    node = pd.read_csv(data_dir+name+'/Nodes.txt', sep='\t', header=0)
    label = torch.tensor(node.iloc[:, 3])
    num_nodes = node.shape[0]
    edge = pd.read_csv(data_dir + name + '/Edges.txt', sep='\t', header=0)
    G = nx.Graph()
    G.add_edges_from(edge.values)
    edge_index = torch.tensor(edge.T.values)
    node_feat = []
    node_feat_d = torch.tensor(pd.read_csv(data_dir+name+'/drug_emb_init.txt', sep='\t', header=None).values, dtype=torch.float32)
    node_feat.append(node_feat_d)
    node_feat_di = torch.tensor(pd.read_csv(data_dir + name + '/disease_emb_init.txt', sep='\t', header=None).values, dtype=torch.float32)
    node_feat.append(node_feat_di)
    node_feat_p = torch.tensor(pd.read_csv(data_dir + name + '/protein_emb_init.txt', sep='\t', header=None).values, dtype=torch.float32)
    node_feat.append(node_feat_p)
    node_feat_pw = torch.tensor(pd.read_csv(data_dir + name + '/pathway_emb_init.txt', sep='\t', header=None).values,
                                dtype=torch.float32)
    node_feat.append(node_feat_pw)

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.adj = G

    return dataset
