import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
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
self_vocab = 

self_mask_span_distribution = None
source = torch.tensor([4, 250004,      3,  16554,  34212, 3324,    4,      3,  76811,      6,  16554,      4,
             5,  23295,      6,  22700, 130425,      4,      2])
line = '[en_XX] [ENT] ▁Romania [TRIPLE] [PRED] ▁description [SUB] ▁Romania [TRIPLE] [PRED] ▁country [SUB] ▁Alba ▁Iulia [TRIPLE]'
add_whole_ent_mask(source, 1)