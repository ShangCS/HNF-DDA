import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'random seed with {seed}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='KEGG')
    parser.add_argument('--embedding_file', type=str, default='emb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./result/')
    
    args = parser.parse_args()
    
    return args

def Xgb_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--n_estimators', type=int, default=950)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=0.9)
    parser.add_argument('--colsample_bytree', type=float, default=0.08)
    
    xgb_args = parser.parse_args()
    
    return xgb_args
    
def split_dataset(dataset, embeddingf, fold, seed):
    d_emb = pd.read_csv('./result/' + dataset + '/emb/' + embeddingf + "_drug.txt", sep='\t', header=None,
                        dtype=np.float32)
    di_emb = pd.read_csv('./result/' + dataset + '/emb/' + embeddingf + "_disease.txt", sep='\t', header=None,
                        dtype=np.float32)
    merged_df = pd.concat([d_emb, di_emb], ignore_index=True)
    embedding_dict = {i: row for i, row in enumerate(merged_df.to_numpy())}
    xs,ys = [],[]
    with open('./data/'+dataset+'/dda.tsv','r') as fin:
        lines = fin.readlines()
        
    for line in lines[1:]:
        line = line.strip().split('\t')
        drug = int(line[0])
        dis = int(line[1])
        label = line[2]
        xs.append(np.concatenate((embedding_dict[drug], embedding_dict[dis]), axis=0))
        ys.append(int(label))

    xs_array = np.array(xs)
    ys_array = np.array(ys)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)

    x_fold, y_fold = [], []

    # 10-fold cross-validation Training set: validation set: test set =8:1:1
    for f, (train_index, test_index) in enumerate(kf.split(xs_array, ys_array), 1):
        x, y = {}, {}
        # Split the training and test sets
        x['train'], x['test'] = xs_array[train_index], xs_array[test_index]
        y['train'], y['test'] = ys_array[train_index], ys_array[test_index]
        
        # Further split the training set into a training set and a validation set
        x['train'], x['valid'], y['train'], y['valid'] = train_test_split(
            x['train'], y['train'], test_size=1/(fold-1),
            random_state=seed, stratify=y['train'])
        x_fold.append(x)
        y_fold.append(y)

    return x_fold, y_fold, embedding_dict

def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score, 
        roc_auc_score, 
        average_precision_score, 
        f1_score
    ] 
    
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list,pred_list))
        else:
            scores.append(metric(target_list, pred_list.round())) 
    return scores

def predict_dda(args, xgb_args, run):
    float2str = lambda x: '%0.4f' % x
    metric_header = ["# Run", "# Fold", "AUROC", "AUPRC", "Accuracy", "F1-score"]
    score_table = PrettyTable(metric_header)
    x_fold,y_fold,embedding_dict = split_dataset(args.dataset, args.embedding_file, args.fold, args.seed)
    clfs, accs, aurocs, auprcs, f1s = [], [], [], [], []
    for f in range(args.fold):
        x = x_fold[f]
        y = y_fold[f]
        clf = XGBClassifier(base_score = 0.5, booster = 'gbtree',eval_metric ='error',objective = 'binary:logistic',
                            gamma = xgb_args.gamma,learning_rate = xgb_args.learning_rate, max_depth = xgb_args.max_depth,
                            n_estimators = xgb_args.n_estimators,tree_method = 'auto',min_child_weight = xgb_args.min_child_weight,
                            subsample = xgb_args.subsample, colsample_bytree = xgb_args.colsample_bytree,scale_pos_weight = 1,
                            max_delta_step = 1,seed = args.seed)
    
        clf.fit(x['train'], y['train'])
    
        preds = {}
        scores = {}
        print(f'Run: {run+1}, Fold: {f+1}')
        for split in ['train','valid','test']:
            preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
            scores[split] = return_scores(y[split], preds[split])
            print(f'{split.upper():5} set | Acc: {scores[split][0]*100:.2f}% | AUROC: {scores[split][1]:.4f} | AUPR: {scores[split][2]:.4f} | F1-score: {scores[split][3]:.4f}')
            if split == 'test':
                accs.append(scores[split][0])
                aurocs.append(scores[split][1])
                auprcs.append(scores[split][2])
                f1s.append(scores[split][3])
                score_lst = ["run " + str(run + 1)] + ["fold " + str(f + 1)] + list(map(float2str, [scores[split][1], scores[split][2], scores[split][0], scores[split][3]]))
                score_table.add_row(score_lst)

        clfs.append(clf)
        print('='*75)
    
    acc_m = sum(accs)/len(accs)
    auroc_m = sum(aurocs)/len(aurocs)
    auprc_m = sum(auprcs)/len(auprcs)
    f1_m = sum(f1s)/len(f1s)
    print(f'Fold mean | Acc: {acc_m*100:.2f}% | AUROC: {auroc_m:.4f} | AUPR: {auprc_m:.4f} | F1-score: {f1_m:.4f}')
    print('='*75)

    score_lst = ["run " + str(run + 1)] + ["fold mean"] + list(map(float2str, [auroc_m, auprc_m, acc_m, f1_m]))
    score_table.add_row(score_lst)
    score_prettytable_file = os.path.join(args.save_dir+args.dataset, "score_table.txt")
    with open(score_prettytable_file, 'a') as fp:
        fp.write(score_table.get_string())
        fp.write('\n')

    return acc_m, auroc_m, auprc_m, f1_m, clfs
          
def trainning(xgb_args, args, random_integers):
    float2str = lambda x: '%0.4f' % x
    clfs, accs, aurocs, auprcs, f1s = [], [], [], [], []
    for r in range(args.run):
        args.seed = random_integers[r]
        acc_r, auroc_r, auprc_r, f1_r, clf_r = predict_dda(args, xgb_args, r)
        accs.append(acc_r)
        aurocs.append(auroc_r)
        auprcs.append(auprc_r)
        f1s.append(f1_r)
        clfs.append(clf_r)
    
    acc_m = sum(accs)/len(accs)
    auroc_m = sum(aurocs)/len(aurocs)
    auprc_m = sum(auprcs)/len(auprcs)
    f1_m = sum(f1s)/len(f1s)

    metric_header = ["# Run", "# Fold", "AUROC", "AUPRC", "Accuracy", "F1-score"]
    score_table = PrettyTable(metric_header)
    for r in range(args.run):
        print(f'Run {str(r+1)} mean | Acc: {accs[r]*100:.2f}% | AUROC: {aurocs[r]:.4f} | AUPR: {auprcs[r]:.4f} | F1-score: {f1s[r]:.4f}')
        score_lst = ["run " + str(r + 1)] + ["fold mean"] + list(map(float2str, [aurocs[r], auprcs[r], accs[r], f1s[r]]))
        score_table.add_row(score_lst)
    print('='*75)
    print(xgb_args)
    print(f'Run mean | Acc: {acc_m*100:.2f}% | AUROC: {auroc_m:.4f} | AUPR: {auprc_m:.4f} | F1-score: {f1_m:.4f}')
    print('='*75)
    score_lst = ["run mean"] + ["mean"] + list(map(float2str, [auroc_m, auprc_m, acc_m, f1_m]))
    score_table.add_row(score_lst)
    
    score_prettytable_file = os.path.join(args.save_dir+args.dataset, "score_table.txt")
    with open(score_prettytable_file, 'a') as fp:
        fp.write(score_table.get_string())
        fp.write('\n')
    print('score table svaed')

    save_path = args.save_dir+args.dataset+'/xgb_model.pkl'
    with open(save_path,'wb') as fw:
        pickle.dump(clfs, fw)
    print(f'saved XGBoost classifier: {save_path}')
    print('='*75)

    return auroc_m, auprc_m, acc_m, f1_m

if __name__ == '__main__':
    float2str = lambda x: '%0.4f' % x
    xgb_args=Xgb_parse_args()
    args=parse_args()
    set_seed(args.seed)
    random_integers = [random.randint(1, 1000) for _ in range(args.run)]
    auroc_m, auprc_m, acc_m, f1_m = trainning(xgb_args, args, random_integers)
