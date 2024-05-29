import torch
import subprocess
import numpy as np

def select_zero_indices(matrix, num, random_seed=None):
    np.random.seed(random_seed)

    # The row-column coordinates of 0 elements are randomly selected as many as there are elements with value 1
    zero_indices = np.column_stack(np.where(matrix == 0))
    selected_zero_indices = zero_indices[np.random.choice(len(zero_indices), num, replace=False)]

    index = matrix.index
    columns = matrix.columns

    negative_list = []

    for x, y in selected_zero_indices:
        negative_x = index[x]
        negative_y = columns[y]
        negative_list.append([negative_x, negative_y])

    return negative_list

def get_negative_subgraph(sub, neg_idx_range):
    """
    Random interference with 50% of the nodes
    """
    temp = sub[-1]
    sub_neg = np.random.permutation(sub[:-1]).tolist()
    for i in range(0, int(len(sub_neg) / 2 + 1)):
        while True:
            neg_idx = np.random.randint(neg_idx_range[0], neg_idx_range[1])
            if neg_idx not in sub:
                sub_neg[i] = neg_idx
                break
    sub_neg.append(temp)

    return sub_neg


def get_meta_paths(graph, start_node, end_node, current_path, all_paths, max_length):
    """
    Get meta paths for graph
    """
    if current_path[-1] == end_node and len(current_path) == max_length:
        all_paths.append(current_path.copy())
        return
    if len(current_path) >= max_length:
        return
    if not graph.has_node(start_node):
        return

    for neighbor in graph.neighbors(current_path[-1]):
        if neighbor not in current_path:
            new_path = current_path + [neighbor]
            get_meta_paths(graph, start_node, end_node, new_path, all_paths, max_length)


def get_subgraphs(adj, drug_idx, disease_idx, neg_idx_range):
    """
    Get positive and negative subgraphs of nodes
    """
    subs_dict = dict()
    single_count = 0

    for i in drug_idx:
        sub_list = []
        for j in disease_idx:
            single_count = single_count + 1
            drug_sub = []
            disease_sub = []
            all_paths = []
            start_node = i
            end_node = j
            max_length = 3
            get_meta_paths(adj, start_node, end_node, [start_node], all_paths, max_length)
            if len(all_paths) != 0:
                for path in all_paths:
                    drug_sub.append(path[1])
                    disease_sub.append(path[1])

                drug_sub.append(end_node)
                drug_sub_neg = get_negative_subgraph(drug_sub, neg_idx_range)
                sub_list.append([drug_sub, drug_sub_neg])

                disease_sub.append(start_node)
                disease_sub_neg = get_negative_subgraph(disease_sub, neg_idx_range)
                temp = [disease_sub, disease_sub_neg]
                if j in subs_dict.keys():
                    value = subs_dict[j]
                    value.append(temp)
                    subs_dict[j] = value
                else:
                    subs_dict[j] = [temp]
        if len(sub_list) != 0:
            subs_dict[i] = sub_list

    return subs_dict

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory