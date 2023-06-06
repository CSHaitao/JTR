
from cProfile import label
import email
import sys
sys.path += ["./"]
import os
import time
import torch
import random
import faiss
import joblib
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl
from construct_tree import TreeInitialize,TreeNode
from torch import nn
from tqdm import tqdm, trange
from numba import jit


@jit(nopython=False)
def candidates_generator(embeddings,node_dict,topk):  #qid
    """layer-wise retrieval algorithm in prediction."""
    root = node_dict['0']
    Q, A = root.children, []
    layer = 0
    embedding = embeddings.reshape(1,768).cpu().numpy()

    while Q:
        layer = layer+1
        B = []
        for node in Q:
            if node.isleaf is True:   #如果是叶节点        
                A.append(node)  
                B.append(node)
        for node in B:
            Q.remove(node)

        if(len(Q) == 0):
            break

        probs = []
        embeddings = []
        for node in Q:
            embeddings.append(node.embedding)

        embeddings =np.array(embeddings)
       
        probs = np.dot(embedding, embeddings.T).reshape(-1,).tolist()
        prob_list = list(zip(Q, probs))
        prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)

        I = []
        if len(prob_list) > topk:
            for i in range(topk):
                I.append(prob_list[i][0])
        else:
            for p in prob_list:
                I.append(p[0])


        Q = []
        while I:
            node = I.pop()
            for child in node.children:
                Q.append(child)

    # A = []
    # for i in range(topk):
    #     A.append(prob_list[i][0].val) 
    
    # return A
    probs = []
    leaf_embeddings = []
    for leaf in A:
        leaf_embeddings.append(leaf.embedding)
    leaf_embeddings =np.array(leaf_embeddings)
        
    probs =  np.dot(embedding, leaf_embeddings.T).reshape(-1,).tolist()
    prob_list = list(zip(A, probs))
    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    A = []
    for i in range(topk):
        A.append(prob_list[i][0].val)  #pid
    return A

@numba.jit(nopython=True)
def metrics_count(embeddings,node_dict,topk):   #(vtest, tree.root, 10, model
    rank_list = []
    size = embeddings.shape[0] 
    for i in range(size):
        cands = candidates_generator(embeddings,node_dict,topk)  #返回的节点
        rank_list.append(cands)
    return rank_list