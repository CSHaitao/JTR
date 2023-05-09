from asyncio.format_helpers import _format_callback_source
import enum
import sys
import random
from grpc import compute_engine_channel_credentials

from torch._C import dtype
sys.path += ['./']
import torch
from torch import nn
import numpy as np
import transformers
if int(transformers.__version__[0]) <=3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaModel
import torch.nn.functional as F
from torch.cuda.amp import autocast


class embedding_model(nn.Module):
    def __init__(self,node_embeddings, label_offest, device):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(node_embeddings,freeze = False).to(device)
        self.label_offest = label_offest
        self.device = device
        self.loss_fct = nn.CrossEntropyLoss().to(device)

    def forward(self, query_embeddings, all_rel_label, node_list, node_dict,label_offest,layer,all_node):
        batch=query_embeddings.shape[0]  
        scores = None
        for i in range(batch):
            features = None
            query_embedding = query_embeddings[i].reshape(-1,768)
            rel_label = all_rel_label[i]    
            offest = label_offest[rel_label]
            node_embedding = self.embedding(offest).reshape(-1,768)
            child_embeddings = node_embedding

            negetive = []
            node_father = node_dict[rel_label].parent
            for node_bro in node_father.children:
                negetive.append(node_bro.val)
            num = len(rel_label)
            tmp_node = node_list[num-1]
            if rel_label in negetive:
                negetive.remove(rel_label)
            index_list = random.sample(range(len(tmp_node)), all_node-1) 

            for i in index_list:
                negetive.append(tmp_node[i])
            if rel_label in negetive:
                negetive.remove(rel_label)

            if len(negetive)>all_node-1:
                negetive = negetive[:all_node-1]

            
            for index in negetive:
                offest = label_offest[index]
                child_embedding = self.embedding(offest).reshape(-1,768)
            
                if child_embeddings is None: 
                    child_embeddings = child_embedding
                else:
                    child_embeddings = torch.cat([child_embeddings, child_embedding], dim=0)
            
            features = torch.matmul(query_embedding, child_embeddings.T)
    
            score = features.reshape(1,-1)
            if(score.shape[1]!= all_node):
                num_len = all_node-score.shape[1]
                other = torch.zeros(1,num_len).to(self.device)
                score = torch.cat((score,other),1)
       

            if scores is None:
                scores = score
            else:
                scores = torch.cat([scores, score], dim=0)
                
                
        return scores

    def get_embedding(self, node_dict, node_list, layer): 
        tmp_node = node_list[layer]
        for index in tmp_node:
            offest = self.label_offest[index]
            node_embedding = np.array(self.embedding(offest).cpu().detach())
            node_dict[index].embedding = node_embedding
        return node_dict

    def predict(self, query_embeddings, node): 
        query_embedding = query_embeddings.reshape(-1,768)
        offest = self.label_offest[node.val]
        node_embedding = self.embedding(offest)
        node_embeddings = torch.tensor(node_embedding).reshape(-1,768).to(self.device)
        score  = torch.matmul(query_embedding, node_embeddings.T)
        return score



class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask):
        # TODO should raise NotImplementedError
        # temporarily do this  
        return None 

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)


class RobertaDot(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :  
            config.return_dict = False
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1



class RobertaDot_4(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :   
            config.return_dict = False
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size*4
        else:
            self.output_embedding_size = config.hidden_size*4
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1



class RobertaDot_InBatch(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids, other_doc_attention_mask,
            rel_pair_mask, hard_pair_mask)


class RobertaDot_Rand(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids, other_doc_attention_mask,
            hard_pair_mask)


def inbatch_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):

    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    batch_size = query_embs.shape[0]
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
   
        single_positive_scores = torch.diagonal(batch_scores, 0)

        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
        if rel_pair_mask is None:
            rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)                
 
        batch_scores = batch_scores.reshape(-1)
        logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                batch_scores.unsqueeze(1)], dim=1)  
     
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)

        first_loss, first_num = loss.sum(), rel_pair_mask.sum()

    if other_doc_ids is None:
        return (first_loss/first_num,)

    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)  

        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
      
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    
    return ((first_loss+second_loss)/(first_num+second_num),)


def randneg_train(query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            hard_pair_mask=None):

    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        single_positive_scores = torch.diagonal(batch_scores, 0)
    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)
    
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                other_batch_scores.unsqueeze(1)], dim=1)  
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    return (second_loss/second_num,)