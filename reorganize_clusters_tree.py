from cProfile import label
from cgi import print_environ
import email
from json import load
from lib2to3.pytree import Node
import sys
from tkinter import Y
from traceback import print_tb
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
from numba import njit
from numba.core import types
from numba.typed import Dict
import scipy.sparse as smat
import pickle as pkl
from construct_tree import TreeInitialize,TreeNode
from torch import nn
from tqdm import tqdm, trange
from model_search_cos import metrics_count
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)



def _define_node_emebedding(arr_node, node_dict):
    for key in arr_node[5]:
        node = node_dict[key]
        if node.isleaf == False:
            # print(node.val)
            embedding = [0 for _ in range(768)]
            num = 0
            for child in node.children:
                embedding = np.sum([embedding,child.embedding],axis=0)
                num += 1
            print(num)
            node.embedding = [ x/num for x in embedding]
        else:
            pass
    for key in arr_node[4]:
        node = node_dict[key]
        if node.isleaf == False:
            embedding = [0 for _ in range(768)]
            num = 0
            for child in node.children:
                embedding = np.sum([embedding,child.embedding],axis=0)
                num += 1
            node.embedding = [ x/num for x in embedding]
    for key in arr_node[3]:
        node = node_dict[key]
        if node.isleaf == False:
            embedding = [0 for _ in range(768)]
            num = 0
            for child in node.children:
                embedding = np.sum([embedding,child.embedding],axis=0)
                num += 1
            node.embedding = [ x/num for x in embedding]
    for key in arr_node[2]:
        node = node_dict[key]
        if node.isleaf == False:
            embedding = [0 for _ in range(768)]
            num = 0
            for child in node.children:
                embedding = np.sum([embedding,child.embedding],axis=0)
                num += 1
            node.embedding = [ x/num for x in embedding]
    for key in arr_node[1]:
        node = node_dict[key]
        if node.isleaf == False:
            embedding = [0 for _ in range(768)]
            num = 0
            for child in node.children:
                embedding = np.sum([embedding,child.embedding],axis=0)
                num += 1
            node.embedding = [ x/num for x in embedding]
    return node_dict






def _node_list(root):
        def node_val(node):
            if(node.isleaf == False):
                return node.val
            else:
                return node.val
            
        node_queue = [root]
        arr_arr_node = []
        arr_arr_node.append([node_val(node_queue[0])])
        while node_queue:
            tmp = []
            tmp_val = []
            for node in node_queue:
                for child in node.children: 
                    tmp.append(child)
                    tmp_val.append(node_val(child))
            if len(tmp_val) > 0:
                arr_arr_node.append(tmp_val)
            node_queue = tmp
        return arr_arr_node

@njit
def construct_new_C_and_Y(
    counts_rows,
    counts_cols,
    counts,
    row_ids,
    row_ranges,
    C_rows,
    sort_idx,
    nr_labels,
    max_cluster_size,
    n_copies,
):
    """Determine the new clustering matrix and the new label matrix given the couting matrix.

    This function implements Eq.(10) in our paper. I.e. given the couting matrix C = Y^T * M, 
    we select the correct cluster id for each label one by one, in descending order of C entries,
    possibly assign a label multiple times (`n_copies`) to different clusters. Finally, the new 
    cluster and new label matrix is returned. Notice that Numba is used here, this prevents us 
    from passing scipy sparse matrix directly.
    
    Args:
        counts_rows, counts_cols, counts: The counting matrix in COO format.
        row_ids, row_ranges: The indices and indptr of original Y matrix in CSC format.
        C_rows: Clustering matrix C in LIL format, converted to list of numpy arrays.
        sort_idx: Index of counts_{rows,cols} to sort them in decending order.
        nr_labels: Number of original labels.
        max_cluster_size: (Unused for now) Hard constraints to limit the number of labels 
            in each cluster (to balance cluster size).
        n_copies: Max number of copies for each label (\lambda in our paper).

    Returns:
        New cluster matrix (`new_C_*`), new label matrix (`new_Y_*`), the replicated label
        assignment (`C_overlap_*`), number of duplicated labels (`nr_copied_labels`), a map 
        from new label id to the underlying label id (`mapper`), unused labels that never 
        show up in training (`unused_labels`), number of lightly used labels (`nr_tail_labels`).
    """
    # construct empty cluster matrix and label matrix
    nr_copied_labels = 0
    new_C_cols = []
    new_C_rows = []
    new_C_data = []
    new_Y_rows = []
    labels_included = set()
    mapper = Dict.empty(key_type=types.int64, value_type=types.int64,)
    cluster_size = Dict.empty(key_type=types.int64, value_type=types.int64,)
    pseudo_label_count = Dict.empty(key_type=types.int64, value_type=types.int64,)
    # results
    C_overlap_rows, C_overlap_cols = [], []
    max_count = n_copies
    # adding labels to clusters one by one in descending frequency
    for idx in sort_idx:
        label_id = counts_rows[idx]   
        leaf_id = counts_cols[idx]    
        if label_id in pseudo_label_count and pseudo_label_count[label_id] >= max_count: 
            continue
        # If you need to contrain the max cluster size, then
        # uncomment following two lines
        # if label_count[leaf_id] >= max_cluster_size:
        #    continue
        if leaf_id not in cluster_size:
            cluster_size[leaf_id] = 1    
        else:
            cluster_size[leaf_id] += 1

        if label_id not in pseudo_label_count:
            pseudo_label_count[label_id] = 1   
        else:
            pseudo_label_count[label_id] += 1

        if label_id in labels_included:  #
            # add a pseudo label that duplicates label_id
            pseudo_label_id = nr_copied_labels + nr_labels
            mapper[pseudo_label_id] = label_id   
            # add one more row to C (in lil format)
            new_C_rows.append(nr_copied_labels)
            new_C_cols.append(leaf_id)
            new_C_data.append(1.0)
            # add one more column to Yt
            examples = row_ids[row_ranges[label_id] : row_ranges[label_id + 1]]
            new_Y_rows.append(examples)
            nr_copied_labels += 1
        else:
            # add a new label  
            labels_included.add(label_id)
            C_overlap_rows.append(label_id)
            C_overlap_cols.append(leaf_id)

        # exit early if we have too many effective labels
        if len(mapper) >= max_count * nr_labels:
            break
    # add missing labels back to clusters
    nr_tail_labels = 0
    for label_id in range(nr_labels):
        if label_id not in labels_included:
            original_leaf_id = C_rows[label_id][0]  #   
            C_overlap_rows.append(label_id)
            C_overlap_cols.append(original_leaf_id)
            labels_included.add(label_id)
            nr_tail_labels += 1

    unused_labels = set()
    for label_id in range(nr_labels):
        if label_id not in labels_included:
            unused_labels.add(label_id)

    # new_Y elements
    new_Y_indptr = [0]
    new_Y_indices = []
    for rows in new_Y_rows:
        new_Y_indptr.append(new_Y_indptr[-1] + len(rows))
        new_Y_indices.extend(rows)
    new_Y_data = np.ones(len(new_Y_indices), dtype=np.int32)
    return (
        np.array(new_C_cols),
        np.array(new_C_rows),
        np.array(new_C_data),
        new_Y_data,
        new_Y_indices,
        new_Y_indptr,
        C_overlap_cols,
        C_overlap_rows,
        nr_copied_labels,
        mapper,
        unused_labels,
        nr_tail_labels,
    )


def get_matching_matrix(rank_output, dict_value):
    path_to_rank = rank_output
    row = []
    col = []
    with open(path_to_rank,'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                labelid = l[1]
                row.append(qid)
                col.append(dict_value[labelid])       
            except:
                raise IOError('\"%s\" is not valid format' % l)

    row = np.array(row)
    print(row.shape)
    col = np.array(col)
    row_num = int(len(row)/5)
    data = np.ones((len(col),))
    data_matrix = smat.csr_matrix((data,(row,col)),shape=(row_num,len(dict_value)))

    return data_matrix

def get_cluster_matrix(leaf_dict,dict_value):
    row = []
    col = []
    for node in leaf_dict.values():
        for pid in node.pids:
            row.append(pid)
            col.append(dict_value[node.val])

    row = np.array(row)
    col = np.array(col)
    data = np.ones((len(col),))
    data_matrix = smat.csr_matrix((data,(row,col)),shape=(len(row),len(dict_value)))
    return data_matrix

def Get_New_C(args):

    leaf_dict = load_object(f'{args.leaf_path}')

    value_dict = {}
    dict_value = {}
    i = 0
    for node in leaf_dict.values():
        value_dict[i] = node.val
        dict_value[node.val] = i
        i = i + 1

    ## get matric M
    path_to_rank = f"{args.M_path}"
    M = get_matching_matrix(path_to_rank,dict_value)
    print("M shape")
    
    print(M.shape) 

    # get matric C
    C = get_cluster_matrix(leaf_dict,dict_value) 
    
    print(C.shape) 


  
    qrel_path = f'{args.Y_path}'
    row = []
    col = []
    data_dev = pd.read_csv(qrel_path, header=None, names=["qid","pid",'rank'], sep='\t')
    for qid,pid in zip(data_dev.qid,data_dev.pid):
        row.append(qid)
        col.append(pid)
    row = np.array(row)
    col = np.array(col)
    data = np.ones((len(col),))
    Y = smat.csr_matrix((data,(row,col)),shape=(M.shape[0],C.shape[0])) 
    print(Y.shape)
    counts = Y.transpose().dot(M).tocoo()
    counts.eliminate_zeros()   
    counts_rows, counts_cols, counts = counts.row, counts.col, counts.data
    print(len(counts))
  
    sort_idx = np.argsort(counts)[::-1] 
   
 
    Yt_csc = Y.tocsc()  
    row_ranges = Yt_csc.indptr   
    row_ids = Yt_csc.indices  

    C = C.tolil()
    C_rows = C.rows   

    max_cluster_size = int(1.0 * C.shape[0] / C.shape[1])
    (
        new_C_cols,
        new_C_rows,
        new_C_data,
        new_Y_data,
        new_Y_indices,
        new_Y_indptr,
        C_overlap_cols,
        C_overlap_rows,
        out_labels,
        mapper,
        unused_labels,
        nr_tail_labels,
    ) = construct_new_C_and_Y(
        np.asarray(counts_rows, dtype=np.int32),
        np.asarray(counts_cols, dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(row_ranges, dtype=np.int32),
        [np.asarray(row, dtype=np.int32) for row in C_rows],
        sort_idx,
        Y.shape[1],
        max_cluster_size,
        args.overlap,
    )
    C_overlap = smat.coo_matrix(
        (np.ones_like(C_overlap_cols), (C_overlap_rows, C_overlap_cols)),
        shape=C.shape,
        dtype=C.dtype,
    ).tocsr()
 
    print(f"#copied labels: {out_labels}, #tail labels: {nr_tail_labels}")
 
    new_C = smat.csr_matrix((new_C_data,(new_C_rows,new_C_cols)),shape=(out_labels, C.shape[1]),dtype=C.dtype)
    C_new = smat.vstack((C_overlap, new_C), format="csc")
    
    new_Y = smat.csc_matrix(
        (new_Y_data, new_Y_indices, new_Y_indptr),
        shape=(Y.shape[0], len(new_Y_indptr) - 1),
        dtype=Y.dtype,
    )
    Y = smat.hstack((Y, new_Y), format="csr")
    smat.save_npz(f'{args.save_dir}/C.npz', C_new, compressed=True)
    smat.save_npz(f'{args.save_dir}/Y.npz', Y, compressed=True)
    smat.save_npz(f'{args.save_dir}/M.npz', M, compressed=True)
    save_object(dict(mapper),f'{args.save_dir}/mapper.pkl')

def tree_update(args): 
    # tree_path = f"{args.raw_tree_path}"
    tree = load_object(args.raw_tree_path)
    # tree_leaf_dict = load_object(args.leaf_path)
    print(len(tree.leaf_dict))
    for leaf in tree.leaf_dict:    ##清空pids
        node = tree.leaf_dict[leaf]
        node.pids = []  
        

    # leaf_dict = load_object(f'{args.leaf_path}')
    value_dict = {}
    dict_value = {}
    i = 0
    for node in tree.leaf_dict.values():
        value_dict[i] = node.val
        dict_value[node.val] = i
        i = i + 1

    fname = f'{args.save_dir}/C.npz'
    C = smat.load_npz(fname).tocoo()

    C.eliminate_zeros()   
    pids, cluster_ids = C.row, C.col
  
    for pid,cluster_id in zip(pids,cluster_ids):
        tree.leaf_dict[value_dict[cluster_id]].pids.append(pid)

    
  
    dict_label = {}   
    for leaf in tree.leaf_dict:
        node = tree.leaf_dict[leaf]
        pids = node.pids
       
        for pid in pids:
            dict_label[pid] = str(node.val)
  
    # print("Update embedding")
    # for leaf in tree.leaf_dict:
    #     node = tree.leaf_dict[leaf]
    #     pids = node.pids
    #     num = 0
    #     embedding = [0 for _ in range(768)]
    #     for pid in pids:
    #         dict_label[pid] = str(node.val)
            
    #         try:
    #             embedding = np.sum([embedding,pid_embeddings_all[pid]],axis=0)
    #             num = num+1
    #         except:
    #             pass
    #     node.embedding = [ x/num for x in embedding]

    root = tree.root
    node_list = _node_list(root)
    save_object(node_list,f'{args.save_dir}/node_list.pkl')

    node_dict = {}
    node_queue = [tree.root]
    while node_queue:
        current_node = node_queue.pop(0) 
        node_dict[current_node.val] = current_node
        for child in current_node.children:
            node_queue.append(child)

 
    save_object(tree.leaf_dict,f'{args.save_dir}/leaf_dict.pkl')
    df = pd.DataFrame.from_dict(dict_label, orient='index',columns=['labels'])
    df = df.reset_index().rename(columns = {'index':'pid'})
    df.to_csv(f'{args.save_dir}/pid_labelid.memmap',header=False, index=False)
    save_object(node_dict,f'{args.save_dir}/node_dict.pkl')

    path = 'dev-qrel.tsv'
    output_file_path = f'{args.save_dir}/dev_label.tsv'
    
    outfile = open(output_file_path,'w')
    with open(path,'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                pid = int(l[2])
                rel = int(l[3])
                label = dict_label[pid][:]
                if rel != 0:
                    outfile.write(f"{qid}\t0\t{label}\t{rel}\n")  
            except:
                raise IOError('\"%s\" is not valid format' % l)   



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--output_dir", type=str, default=f"../tree/doc/new_tree")
    parser.add_argument("--type", choices=["doc", "passage"], default="doc")
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--raw_tree_path", type=str, default=f"tree.pkl")
    parser.add_argument("--leaf_path", type=str, default=f"leaf_dict.pkl")
    parser.add_argument("--M_path", type=str, default=f"recall_train_5.tsv")
    parser.add_argument("--Y_path", type=str, default=f"train.rank_100.tsv")

    args = parser.parse_args()
    args.save_dir = f"{args.output_dir}/{args.overlap}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    Get_New_C(args)
    tree_update(args)

   