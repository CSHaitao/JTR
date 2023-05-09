# Constructing Tree-based Index for Efficient and Effective Dense Retrieval

The official repo for our SIGIR'23 Full paper: [Constructing Tree-based Index for Efficient and Effective Dense Retrieval](https://arxiv.org/abs/2304.11943)

## Introduction

To balance the effectiveness and efficiency of the tree-based indexes, we propose **JTR**, which stands for **J**oint optimization of **TR**ee-based index and query encoding. To jointly optimize index structure and query encoder in an end-to-end manner, JTR drops the original ``encoding-indexing" training paradigm and designs a unified contrastive learning loss. However, training tree-based indexes using contrastive learning loss is non-trivial due to the problem of differentiability. To overcome this obstacle, the tree-based index is divided into two parts: cluster node embeddings and cluster assignment. For differentiable cluster node embeddings, which are small but very critical, we design tree-based negative sampling to optimize them. For cluster assignment, an overlapped cluster method is applied to iteratively optimize it.



## Preprocess


## Tree Initialization


## Train


## Reorganize Cluster


## 















## Citations

If you find our work useful, please do not save your star and cite our work:

```
@misc{SAILER,
      title={SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval}, 
      author={Haitao Li and Qingyao Ai and Jia Chen and Qian Dong and Yueyue Wu and Yiqun Liu and Chong Chen and Qi Tian},
      year={2023},
      eprint={2304.11370},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```