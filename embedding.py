import os
import copy
import pickle
import random
import argparse
import warnings
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset
from parse import parse_method, parser_add_main_args
from data_utils import adj_mul, get_gpu_memory_map, get_subgraphs
from torch_geometric.utils import remove_self_loops, add_self_loops

warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args, dataset, c, d, device, feat_list, adjs, subs_dict, drug_idx, disease_idx):
    print(args)
    fix_seed(args.seed)

    ### Load method ###
    model = parse_method(args, c, d, device)

    criterion = nn.NLLLoss()

    model.train()

    ### Training loop ###
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    best_loss = loss_last = float('inf')
    epochs_without_improvement = 0
    best_emb = None

    print("Trainging...")
    print('MODEL:', model)
    loss = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        emb, out, link_loss_, sub_loss_ = model(feat_list, adjs, subs_dict, args.tau)

        out = F.log_softmax(out, dim=1)
        loss = criterion(out, dataset.label.squeeze(1))
        class_loss = loss.item()

        loss = loss + (-args.lamda * sum(link_loss_) / len(link_loss_)) - (args.beta * sum(sub_loss_) / len(sub_loss_))
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch:02d}, '
              f'Sum_Loss: {loss:.4f}, '
              f'Class_Loss: {class_loss:.4f}, '
              f'Link_Loss: {-sum(link_loss_) / len(link_loss_):.4f}, '
              f'Sub_Loss: {-sum(sub_loss_) / len(sub_loss_):.4f}. ')

        if torch.isinf(loss).any():
            print("Loss is infinity. Training stopped.")
            break
        if torch.isnan(loss):
            print("Loss is NaN. Training stopped.")
            break

        loss_abs = abs(loss.item() - loss_last)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(model)
            best_emb = emb
        if loss_abs < 1e-3:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        # 判断是否进行 Early Stopping
        if epochs_without_improvement >= 20:
            print("Early stopping after {} epochs without improvement.".format(epochs_without_improvement))
            break
        loss_last = loss.item()
    if args.save_model:
        torch.save(best_model.state_dict(), './result/' + args.dataset + '/emb_model.pkl')
        print("Model saved.")
    emb_run = best_emb.detach().cpu()
    path_drug = './result/' + args.dataset + '/emb/emb_drug'
    path_disease = './result/' + args.dataset + '/emb/emb_disease'
    emb_run_pd = pd.DataFrame(emb_run.numpy())
    emb_run_pd_drug = emb_run_pd.iloc[drug_idx, :]
    emb_run_pd_disease = emb_run_pd.iloc[disease_idx, :]
    emb_run_pd_drug.to_csv(path_drug + '.txt', sep='\t', header=False, index=False)
    emb_run_pd_disease.to_csv(path_disease + '.txt', sep='\t', header=False, index=False)
    print("Embedding saved")

if __name__ == '__main__':
    ### Parse args ###
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### Load and preprocess data ###
    dataset = load_dataset(args.data_dir, args.dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = []
    for feat in dataset.graph['node_feat']:
        d.append(feat.shape[1])

    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    feat_list = []
    for feat in dataset.graph['node_feat']:
        feat_list.append(feat.to(device))

    ### Get nodes list of subgraphs in drug and disease nodes
    drug_idx = []
    disease_idx = []
    neg_idx_range = [0, 0]
    for i in range(dataset.label.shape[0]):
        if dataset.label[i, 0] == 0:
            drug_idx.append(i)
        elif dataset.label[i, 0] == 1:
            disease_idx.append(i)
    neg_idx_range[0] = len(drug_idx) + len(disease_idx)
    neg_idx_range[1] = n - 1

    ### get positive and negative subgraph of target nodes
    '''
    subs_dict = get_subgraphs(dataset.adj, drug_idx, disease_idx, neg_idx_range)

    subs_dict_file = os.path.join(args.data_dir + args.dataset, "subs_dict.pkl")
    with open(subs_dict_file, 'wb') as f:
        pickle.dump(subs_dict, f)
    print("Get subgraph finished")
    '''
    path = args.data_dir + args.dataset + '/subs_dict.pkl'
    with open(path, 'rb') as file:
        subs_dict = pickle.load(file)
    file.close()

    ### Adj storage for relational bias ###
    adjs = []
    adj, _ = remove_self_loops(dataset.graph['edge_index'])
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)

    for i in range(args.rb_order - 1):  # edge_index of high order adjacency
        adj = adj_mul(adj, adj, n)
        adjs.append(adj)
    dataset.graph['adjs'] = adjs

    main(args, dataset, c, d, device, feat_list, dataset.graph['adjs'], subs_dict, drug_idx, disease_idx)
