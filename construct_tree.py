from http.client import PROXY_AUTHENTICATION_REQUIRED
import numpy as np
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'



def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

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

class TreeNode(object):
    """define the tree node structure."""
    def __init__(self, x ,item_embedding = None, layer = None):
        self.val = x   
        self.embedding = item_embedding  
        self.parent = None
        self.children = []
        self.isleaf = False
        self.pids = []
        self.layer = layer
    
    def getval(self):
        return self.val
    def getchildren(self):
        return self.children
    def add(self, node):
            ##if full
        if len(self.children) == 10:
            return False
        else:
            self.children.append(node)
 

class TreeInitialize(object):
    """"Build the random binary tree."""
    def __init__(self, pid_embeddings, pids, blance_factor, leaf_factor):  
        self.embeddings = pid_embeddings
        self.pids = pids
        self.root = None
        self.blance_factor = blance_factor
        self.leaf_factor = leaf_factor
        self.leaf_dict = {}
        self.node_dict = {}
        self.node_size = 0
        
    def _k_means_clustering(self, pid_embeddings): 
        if len(pid_embeddings)>1000000:
            idxs = np.arange(pid_embeddings.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[0:1000000]
            train_embeddings = pid_embeddings[idxs] 
        else:
            train_embeddings = pid_embeddings
        train_embeddings = pid_embeddings
        kmeans = KMeans(n_clusters=self.blance_factor, max_iter=3000, n_init=100).fit(train_embeddings)
        return kmeans

    def _build_ten_tree(self, root, pid_embeddings, pids, layer):
        if len(pids) < self.leaf_factor:
            root.isleaf = True
            root.pids = pids
            self.leaf_dict[root.val] = root
            return root

        kmeans = self._k_means_clustering(pid_embeddings)
        clusters_embeddings = kmeans.cluster_centers_
        labels = kmeans.labels_
        for i in range(self.blance_factor): ## self.blance_factor < 10
            val = root.val + str(i)
            node = TreeNode(x = val, item_embedding=clusters_embeddings[i],layer=layer+1)
            node.parent = root
            index = np.where(labels == i)[0]
            pid_embedding = pid_embeddings[index]
            pid = pids[index]
            node = self._build_ten_tree(node, pid_embedding, pid, layer+1)
            root.add(node)
        return root

    def clustering_tree(self):  
        root = TreeNode('0')
        self.root = self._build_ten_tree(root, self.embeddings, self.pids, layer = 0)
        return self.root



    
if __name__ == '__main__':

    type = "passage"
    max_pid = 1000
    pass_embedding_dir = f'/{type}/star/passages.memmap' 
    

    ## build tree
    output_path = f"../tree/{type}/cluster_tree"
    tree_path = f"{output_path}/tree.pkl"
    dict_label = {}
    pid_embeddings_all = np.memmap(pass_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    pids_all = [x for x in range(pid_embeddings_all.shape[0])]
    pids_all = np.array(pids_all)
    tree = TreeInitialize(pid_embeddings_all, pids_all)
    _ = tree.clustering_tree()
    save_object(tree,tree_path)
  

    ## save node_dict
    tree = load_object(tree_path)
    node_dict = {}
    node_queue = [tree.root]
    val = []
    while node_queue:
        current_node = node_queue.pop(0) 
        node_dict[current_node.val] = current_node
        for child in current_node.children:
            node_queue.append(child)
    print("node dict length")
    print(len(node_dict))
    print("leaf dict length")
    print(len(tree.leaf_dict))
    save_object(node_dict,f"{output_path}/node_dict.pkl")

    ## save node_list
    tree = load_object(tree_path)
    root = tree.root
    node_list = _node_list(root)
    save_object(node_list,f"{output_path}/node_list.pkl")


    ## pid2cluster
    for leaf in tree.leaf_dict:
        node = tree.leaf_dict[leaf]
        pids = node.pids
        for pid in pids:
            dict_label[pid] = str(node.val)
    df = pd.DataFrame.from_dict(dict_label, orient='index',columns=['labels'])
    df = df.reset_index().rename(columns = {'index':'pid'})
    df.to_csv(f"{output_path}/pid_labelid.memmap",header=False, index=False)
    
    print('end')
    tree = load_object('tree.pkl')
    print(len(tree.leaf_dict))
    save_object(tree.leaf_dict,'leaf_dict.pkl')
    



            
        

                
          

