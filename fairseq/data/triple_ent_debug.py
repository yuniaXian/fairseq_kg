import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
from torch._C import DisableTorchFunction, Size
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf



class Kg2textEmbedding:

    def __init__(self, dictionary) -> None:
        """ 
        ent
        triple
        sub
        prep
        kg
        text
        """
        
        self.dict = dictionary
        self.eos = self.dict.eos()
        self.bos = self.dict.bos()
        self.ent = self.dict.index("[ENT]")
        self.triple = self.dict.index("[TRIPLE]")
        self.sub = self.dict.index("[SUB]")
        self.pred = self.dict.index("[PRED]")
        self.kg = self.dict.index("[KG]")
        self.lang = self.dict.index("[en_XX]")
        self.text = self.dict.index("[TEXT]")
        self.mask = self.dict.index("<mask>")


    def get_intervals(self, tag_s, tag_t, source):
        # get intervals index of all nearest pair tag_s, tag_t (tokens included)
        s_inds = (source==tag_s).nonzero(as_tuple=True)[0]
        t_inds = (source==tag_t).nonzero(as_tuple=True)[0]
        mask = (t_inds>s_inds.unsqueeze(1))
        inds_matrix = torch.arange(mask.shape[1], 0, -1)
        inds_masked = mask*inds_matrix
        inds = torch.argmax(inds_masked, 1)
        
        paired_t_inds = t_inds[inds]

        mask2 = (s_inds<paired_t_inds)
        intervals = torch.stack([s_inds, paired_t_inds], 1)
        intervals = intervals[mask2]

        return intervals


    def get_one_triple_embedding_kgpt(self, source, embedding):
        # "[ENT] ▁Romania [TRIPLE] [PRED] ▁description [SUB] ▁Romania [TRIPLE] [PRED] ▁country [SUB] ▁Alba ▁Iulia [TRIPLE]"
        ent_inds = (source == self.ent).nonzero(as_tuple=True)[0]
        pred_triple_intervals = self.get_intervals(self.pred, self.triple, source)
        
        first_pred_index = pred_triple_intervals[0][0]
        # fill 1s between [ENT] and [TRIPLE], [TRIPLE] not included
        embedding[range(ent_inds[0], first_pred_index-1)] = 1
        embedding[first_pred_index-1] = 2

        i = 0
        for s, t in pred_triple_intervals:
            embedding[range(s,t+1)] = i+2
            i += 1
        return embedding


    def get_triples_embedding_kgpt(self, source):
        
        ent_inds = (source == self.ent).nonzero(as_tuple=True)[0]

        embedding = torch.zeros(source.shape)
        if ent_inds.size(0) == 1:
            s, t = ent_inds[0], embedding.size(0)
        
        elif ent_inds.size(0) > 1:
            for i in range(ent_inds.size(0)):
                s, t = int(ent_inds[i]), ent_inds[i+1] if i+1<ent_inds.size(0) else source.size(0)-1
                self.get_one_triple_embedding_kgpt(source[s : t], embedding[s : t])
        else:
            return torch.tensor([], dtype=source.dtype)

        return embedding

    def get_entity_embedding_kgpt(self, source):
        # TODO only with one ent_inds
        ent_inds = (source == self.ent).nonzero(as_tuple=True)[0]
        embedding = torch.zeros(source.shape)

        if ent_inds.size(0) == 1:
            embedding[ent_inds[0]:] = 1
        elif ent_inds.size(0) > 1:
            for i in range(ent_inds.size(0)):
                s, t = ent_inds[i], ent_inds[i+1] if i+1<ent_inds.size(0) else source.size(0)-1
                embedding[s : t] = i + 1
        else:
            return torch.tensor([], dtype=source.dtype)

        return embedding

    def get_position_embedding(self, s, t, source):
       
        embedding = torch.zeros(source.shape)
        embedding[s:t+1]= torch.arange(1, t-s+2, dtype=source.dtype)
        return embedding

    def get_embeddings_kgpt(self, source):

        ent_ind = (source == self.ent).nonzero(as_tuple=True)[0]
        first_ent_ind = ent_ind[0]
        last_triple_ind = (source == self.triple).nonzero(as_tuple=True)[0][-1]
        entity_ids = self.get_entity_embedding_kgpt(source)
        triple_ids = self.get_triples_embedding_kgpt(source)
        position_ids = self.get_position_embedding(first_ent_ind, last_triple_ind, source)
        assert source.size(0) == entity_ids.size(0) == triple_ids.size(0) == position_ids.size(0) 

        return {
            "input_ids": source,
            "entity_ids": entity_ids,
            "triple_ids": triple_ids,
            "position_ids": position_ids
        }

    def add_whole_propery_mask(self, source, p, mask_tags=False):
        # determine where is ent part
        # [TRIPLE] [ENT]... [PRED]...[SUB] ...[TRIPLE]

        intervals = self.get_intervals(self.ent, self.pred, source)
        # TODO: check return dtype

        if mask_tags == False:
            intervals[:,0] = intervals[:,0] + 1
            #intervals[:,1] = intervals[:, 1] + 1
        num_to_mask = int(math.ceil(intervals.size(0)* p))
        if num_to_mask == 0:
            return source

        # TODO remove these debug settings
        self.mask_span_distribution = None
        self.replace_length = 1
        if self.mask_span_distribution is not None:
            raise NotImplementedError
        else:
            intervals_to_mask = intervals[torch.randperm(intervals.size(0))[:num_to_mask]]
            #res = map(torch.arange(), intervals[intervals_indices])
            indices_to_mask = torch.tensor([], dtype = intervals.dtype)
            for interval in intervals_to_mask:
                indices_to_mask = torch.cat((indices_to_mask, torch.arange(*interval)), 0)

        source_length = source.size(0)
        to_keep = torch.ones(source_length, dtype=torch.bool)

        if self.replace_length == 0:
            to_keep[indices_to_mask] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices_to_mask] = self.mask
            """
            source[indices[mask_random]] = torch.randint(
                1, len(self_vocab), size=(mask_random.sum(),)
            )
            """
        
        print("Done")


    def add_tag_mask_only(self, source, p, tag):
        indices = (source==tag).nonzero(as_tuple=True)[0]
        num_to_mask = int(math.ceil(indices.size(0)* p))
        if num_to_mask == 0:
            return source
        
        self.mask_span_distribution = None
        self.replace_length = 1
        if self.mask_span_distribution is not None:
            raise NotImplementedError
        else:
            indices_to_mask = indices[torch.randperm(indices.size(0))[:num_to_mask]]
            #res = map(torch.arange(), indices[intervarls_indices])

        source_length = source.size(0)
        to_keep = torch.ones(source_length, dtype=torch.bool)

        if self.replace_length == 0:
            to_keep[indices_to_mask] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices_to_mask] = self.mask
            """
            source[indices[mask_random]] = torch.randint(
                1, len(self_vocab), size=(mask_random.sum(),)
            )
            """
        
        print("Done")

triples = "[en_XX] [ENT] ▁Romania [TRIPLE] [PRED] ▁description [SUB] ▁Romania [TRIPLE] [PRED] ▁country [SUB] ▁Alba ▁Iulia [TRIPLE] [ENT] ▁Romania [TRIPLE] [PRED] ▁description [SUB] ▁Romania [TRIPLE] [PRED] ▁country [SUB] ▁Alba ▁Iulia [TRIPLE]"

from fairseq.data.dictionary import Dictionary
if __name__=="__main__":
    #tgt_dict = Dictionary.load("/media/MyDataStor1/jxian/efs-storage/tokenizer/mbart50/dict/dict.mbart50_wtags.txt")
    tgt_dict = Dictionary.load("/home/ubuntu/efs-storage/tokenizer/mbart50/dict/dict.mbart50_wtags.txt")
    emb = Kg2textEmbedding(tgt_dict)
    src_tokens = tgt_dict.encode_line(triples, append_eos=True, add_if_not_exist=False)

    res = emb.add_tag_mask_only(src_tokens, 0.7, emb.ent)
    res = emb.add_whole_ent_mask_one_triple(src_tokens, 1, mask_tags=True)
    embeddings = emb.get_embeddings_kgpt(src_tokens)

    print(1)



def add_whole_ent_mask(source, p):
    # determine where is ent part
    source_lst = source.tolist()
    ent_starts = (source == self_ent).nonzero(as_tuple=True)[0].tolist()
    #ent_ends = [(source[s:] == self_triple).nonzero(as_tuple=True)[0][0] for s in ent_starts]
    ent_ends = [source_lst.index(self_triple, s) for s in ent_starts]
    assert len(ent_starts) == len(ent_ends)
    ent_starts_ends = torch.tensor([range(ent_starts[i]+1, ent_ends[i]) for i in range(len(ent_starts))])
    # debug TOOD

    #ent_to_data_index = np.array(ent_starts_ends, dtype=np.compat.long),  # starting offset within starting index
    ent_lst = torch.tensor(range(len(ent_starts)))
    num_to_mask = int(math.ceil(ent_lst.size(0)* p))
    if num_to_mask == 0:
        return source

    if self_mask_span_distribution is not None:
        raise NotImplementedError
    else:
        indices = ent_lst[torch.randperm(ent_lst.size(0))[:num_to_mask]]
    source_length = source.size(0)
    to_keep = torch.ones(source_length, dtype=torch.bool)
    if self_replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        source[ent_starts_ends[indices]] = self_mask_idx
        """
        source[indices[mask_random]] = torch.randint(
            1, len(self_vocab), size=(mask_random.sum(),)
        )
        """
    
    print("Done")

    

self_ent = 3 
self_triple = 4
self_pred = 5
self_sub = 6
self_replace_length = -1
self_mask_idx = 250053
mask_random = torch.tensor([])
self_vocab = None

self_mask_span_distribution = None
source = torch.tensor([4, 250004,      3,  16554,  34212, 3324,    4,      3,  76811,      6,  16554,      4,
             5,  23295,      6,  22700, 130425,      4,      2])
line = '[en_XX] [ENT] ▁Romania [TRIPLE] [PRED] ▁description [SUB] ▁Romania [TRIPLE] [PRED] ▁country [SUB] ▁Alba ▁Iulia [TRIPLE]'
add_whole_ent_mask(source, 1)