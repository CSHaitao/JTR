
# Constructing Tree-based Index for Efficient and Effective Dense Retrieval

The official repo for our SIGIR'23 Full paper: [Constructing Tree-based Index for Efficient and Effective Dense Retrieval](https://arxiv.org/abs/2304.11943)

## Introduction

To balance the effectiveness and efficiency of the tree-based indexes, we propose **JTR**, which stands for **J**oint optimization of **TR**ee-based index and query encoding. To jointly optimize index structure and query encoder in an end-to-end manner, JTR drops the original ``encoding-indexing" training paradigm and designs a unified contrastive learning loss. However, training tree-based indexes using contrastive learning loss is non-trivial due to the problem of differentiability. To overcome this obstacle, the tree-based index is divided into two parts: cluster node embeddings and cluster assignment. For differentiable cluster node embeddings, which are small but very critical, we design tree-based negative sampling to optimize them. For cluster assignment, an overlapped cluster method is applied to iteratively optimize it.

![image](./figure/overflow.pdf)

## Preprocess

JTR initializes the document embeddings with STAR, refer to [DRhard](https://github.com/jingtaozhan/DRhard) for details.


Run the following codes in DRhard to preprocess document.
``
python preprocess.py --data_type 0; python preprocess.py --data_type 1
``


## Tree Initialization

After getting the text embeddings, we can initialize the tree using recursive k-means.

Run the following codes:
``
python construct_tree.py
``
We will get the following files:

tree.pkl: Tree structure
node_dict.pkl: Map of node id to node
node_list: Nodes per level
pid_labelid.memmap: Mapping of document ids to clustering nodes
leaf_dict.pkl: Leaf Nodes


## Train
Run the following codes:
``
python train.py --task train
``

The training process trains both the query encoder and the clustering node embeddings. Therefore, we need to save both the node embeddings and the query encoder.

## Inference

Run the following codes:
``
python train.py --task dev
``

The inference process can construct the matrix M for Reorganize Cluster.

## Reorganize Cluster

Run the following codes:
``
python reorganize_clusters_tree.py
``

The re-clustering requires M and Y matrices. Y matrix is constructed by running other retrieval models. M matrix is constructed by inference on the tree index.


## Other
This work was done when I was a beginner and the code was embarrassing. If somebody can further organize and optimize the code or integrate it into Faiss with C. I would appreciate it.

## Citations

If you find our work useful, please do not save your star and cite our work:

```
@misc{JTR,
      title={Constructing Tree-based Index for Efficient and Effective Dense Retrieval}, 
      author={Haitao Li and Qingyao Ai and Jingtao Zhan and Jiaxin Mao and Yiqun Liu and Zheng Liu and Zhao Cao},
      year={2023},
      eprint={2304.11943},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```