from cProfile import label
import email
from lib2to3.pytree import Node
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

from model_search_cos import metrics_count
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
    RobertaConfig)

from dataset import TextTokenIdsCache, SequenceDataset, load_rel, pack_tensor_2D
from model import RobertaDot,embedding_model

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)    


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)


def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))




class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.reldict[item]
        return ret_val


def get_collate_function(mode, max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        
        qids = [x['id'] for x in batch]
        if(mode == 'train'):
            all_rel_pids = [x["rel_ids"] for x in batch]
            return data, qids, all_rel_pids
        else:
            return data, qids
    return collate_function  
    

gpu_resources = []


def get_kmeans_labels(all_rel_pids, pid_label_dict,layer):
    all_rel_labels = []
    for pids in all_rel_pids:
        labels = pid_label_dict[pids[0]][:layer+1]
        all_rel_labels.append(labels)
    return all_rel_labels


def rel2label(all_rel_label):
    labels = []
    for rel in all_rel_label:
        label = int(rel[-1:])
        labels.append(label)
    return labels

def train(args, model):
    """ Train the model """
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    tb_writer = SummaryWriter(os.path.join(args.log_dir, 
        time.strftime("%b-%d_%H:%M:%S", time.localtime())))



    label_path = f"{args.tree_dir}/pid_labelid.memmap"
    pid_label = pd.read_csv(label_path,header=None,names = ["pids","labels"], sep=',',dtype = {'pids':np.int32,'labels':str})
    pid_label_dict = dict(zip(pid_label['pids'], pid_label['labels']))
    
    
    node_dict = load_object('{args.tree_dir}/node_list.pkl')
    node_list = load_object(f"{args.tree_dir}/node_list.pkl")

    
    node_embeddings = []
    label_offest = {}
    j = 0
    for i in node_dict:
        if(i == '0'):
            continue
        label_offest[i] = torch.tensor(int(j), dtype=torch.long).to(args.model_device)
        j = j + 1
        embedding = node_dict[i].embedding
        node_embeddings.append(embedding)
    node_embeddings = np.array(node_embeddings).astype(float)
    node_embeddings = torch.FloatTensor(node_embeddings).to(args.model_device)

    embedding = embedding_model(node_embeddings, label_offest, args.model_device).to(args.model_device)
    args.train_batch_size = args.per_gpu_batch_size
    train_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "train-query"),
        os.path.join(args.preprocess_dir, "train-qrel.tsv"),
        args.max_seq_length
    )
    train_sampler = RandomSampler(train_dataset) 
    collate_fn = get_collate_function(args.task, args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': embedding_model.parameters(), 'lr':0.00001}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    embedding.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)  
    loss_fct = nn.CrossEntropyLoss().to(args.model_device)

  
    negative_num = 8

    for epoch_idx, _ in enumerate(train_iterator):

        for layer in args.layer:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, (batch, qids, all_rel_pids) in enumerate(epoch_iterator):
                batch = {k:v.to(args.model_device) for k, v in batch.items()}

                model.train()          
                query_embeddings = model(          
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], 
                    is_query=True)
            
                all_rel_label = get_kmeans_labels(all_rel_pids, pid_label_dict, args.layer)
            
                scores = embedding(query_embeddings, all_rel_label, node_list, node_dict, label_offest, layer,all_node)
                labels = torch.zeros(len(scores),).to(args.model_device).long()
                loss = loss_fct(scores, labels)

                loss.backward(retain_graph=True)
                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  
                    model.zero_grad()
                    embedding.zero_grad()
                    global_step += 1
                

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                        tb_writer.add_scalar('train/loss', cur_loss, global_step)
                        logging_loss = tr_loss


        
        save_model(model, args.model_save_dir, 'epoch-model_{}-{}'.format(args.layer,epoch_idx+1), args)
        save_dir = os.path.join(args.model_save_dir,'node_dict_{}-{}.pkl'.format(args.layer,epoch_idx+1))
        node_dict = mlp_model.get_embedding(node_dict,node_list,args.layer)  
        save_object(node_dict,save_dir)
        
def evaluate(args, model, node_dict,mode,prefix):
    eval_output_dir = args.eval_save_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    dev_dataset = SequenceDataset(
                    TextTokenIdsCache(args.preprocess_dir, "dev-query"), 
                    64)



    args.eval_batch_size = args.per_gpu_eval_batch_size 
    collate_fn = get_collate_function(mode=mode, max_seq_length = args.max_seq_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_time = 0
    num = 0
    output_file_path = f"{eval_output_dir}/recall_{mode}_{args.topk}.tsv"
    with open(output_file_path, 'w') as outputfile:
        for batch, qids in tqdm(dev_dataloader, desc="Evaluating"):
            model.eval()

            with torch.no_grad():
                batch = {k:v.to(args.model_device) for k, v in batch.items()}
                embeddings = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], 
                    is_query=True)
               
                num = num + 1

                scores = metrics_count(embeddings, node_dict, args.topk)

                for qid, score_one in zip(qids, scores):
                    index = 0
                    for score in score_one:
                        index = index + 1
                        outputfile.write(f"{qid}\t{score}\t{index}\n")
    

    print("num %f" %(num))


def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", choices=["train", "dev"], required=True)
    parser.add_argument("--output_dir", type=str, default=f"output")
    parser.add_argument("---init_path", type=str, default=f"model")
    parser.add_argument("--pembed_path", type=str, default=f"/passages.memmap")
    parser.add_argument("--preprocess_dir", type=str,default=f'/preprocess')
    parser.add_argument("--tree_dir", type=str,default=f'/tree')
    parser.add_argument("--per_gpu_batch_size", type=int, default=32)
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--warmup_steps", default=2000, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--no_cuda", action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)  
    parser.add_argument("--eval_ckpt", type=int, default=5000)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.eval_save_dir = f"{args.output_dir}/eval_results"

    return args

    



def main():
    args = run_parse_args()

    # Setup CUDA, GPU 
    args.model_device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)

    # Set seed
    set_seed(args)

    logger.info(f"load from {args.init_path}")
    if args.task == "train":
        config = RobertaConfig.from_pretrained(args.init_path)
        model = RobertaDot.from_pretrained(args.init_path, config=config)
        model.to(args.model_device)
    logger.info("Training/evaluation parameters %s", args)
    
    
    if args.task == "train":
        os.makedirs(args.model_save_dir, exist_ok=True) 
        train(args, model)
    else:
        result = evaluate(args, model, node_dict, args.task, prefix=f"ckpt-{args.eval_ckpt}")

    

if __name__ == "__main__":
    main()
