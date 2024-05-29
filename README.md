# HNF-DDA
## Paper "HNF-DDA: Subgraph contrastive-driven transFormer-style Heterogeneous Network embedding for Drug-Disease Association prediction"

## Dependencies
HNF-DDA is tested on Ubuntu 22.04 with Python 3.8.

## Code requirements
  - `python`= 3.8
  - `networkx`= 3.1
  - `numpy`=1.24.3
  - `pandas`=2.0.3
  - `scipy`=1.10.1
  - `tqdm`=4.65.0
  - `xgboost`=1.7.6
  - `torch`==1.9.0
  - `scikit-learn`=1.3.0
  - `torch_geometric`==1.7.2
  - `torch_scatter`==2.1.2
  - `torch_sparse`==0.6.18

## Code
- data_utils.py: data pre-processing
- dataset.py: dataset loading
- parse.py: initializing the model parameters
- hnformer: encoder (The base encoder code is derived from [NodeFormer](https://github.com/qitianwu/NodeFormer))
- embedding.py: learning embeddings by hnformer model
- predict_associations.py: prediction drug-disease associations

## Dataset
#### Initial embeddings in the KEGG and HetioNet datasets are available by contacting the authors.


## Tutorial

### Learning embeddings of drug and disease
```
python embedding.py
```

### Drug-Disease Association predication
```
python predict_associations.py
```

## Citation
...
