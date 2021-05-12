# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import fairseq.checkpoint_utils
import numpy as np
import logging
import math
from typing import Dict, List, Optional, Tuple
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
#from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerModel, FairseqEncoderDecoderModel, FairseqEncoder, \
    FairseqIncrementalDecoder, Embedding, TransformerDecoder
#from fairseq.models.bart.model import BARTModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.models.bart.hub_interface import BARTHubInterface

from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable
import pickle
from fairseq.models.bart import BARTModel
from fairseq.data import encoders
from transformers.utils.dummy_pt_objects import MODEL_FOR_CAUSAL_LM_MAPPING
#import os
#import Model
from myutils import *
from torch.nn import Parameter
from kg2textConfig import *

logger = logging.getLogger(__name__)

def save_cfg(loaded):
    with open("config/loaded", "wb") as f:
        pickle.dump(loaded, f)
        f.flush()
        assert f.seek(0) == 0

class Beam(object):
    ''' Beam search '''

    def __init__(self, size, pad_idx, sos_idx, eos_idx, device=False):

        self.size = size
        self._done = False
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.pad_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.sos_idx
        self.finished = [False for _ in range(size)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        print(self.prev_ks)  # TODO check
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        for i in range(self.size):
            if self.finished[i]:
                word_prob[i, :].fill_(-1000)
                word_prob[i, self.pad_idx].fill_(0)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        # best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores  # TODO warning: acts differently between gpus/cpus

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        # if self.next_ys[-1][0].item() == Constants.EOS:
        #    self._done = True
        #    self.all_scores.append(self.scores)
        self.finished = []
        for i in range(self.size):
            self.finished.append(self.next_ys[-1][i].item() in [self.eos_idx, self.pad_idx])

        if all(self.finished):
            self._done = True
        # self._done = self.finished[0]

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.sos_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        if torch.is_tensor(k) and k.dtype != torch.int64:
            k = k.to(torch.int64)
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class KnowledgeEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, cfg: KgptConfig, word_emb): # config = knowledge.json
        """
        {
          "vocab_size": 50264,
          "pad_token_id": 50263,
          "max_entity_embeddings": 30,
          "max_triple_embeddings": 20,
          "max_position_embeddings": 1024,
          "hidden_size": 768,
          "layer_norm_eps": 1e-12,
          "hidden_dropout_prob": 0.1,
        }
        """
        super(KnowledgeEmbeddings, self).__init__()
        if word_emb:
            self.word_embeddings = word_emb
        else:
            self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.encoder_embed_dim, padding_idx=cfg.pad_token_id)
        self.entity_embeddings = nn.Embedding(cfg.max_entity_embeddings, cfg.encoder_embed_dim)
        self.triple_embeddings = nn.Embedding(cfg.max_triple_embeddings, cfg.encoder_embed_dim)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.encoder_embed_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(cfg.encoder_embed_dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.encoder_layerdrop)

    def forward(self, src_tokens):
        input_ids, entity_ids, triple_ids, position_ids = src_tokens[0], src_tokens[1], src_tokens[2], src_tokens[3]
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1] #??
        """
        inputs_embeds = self.word_embeddings(input_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)
        triple_embeddings = self.triple_embeddings(triple_ids)
        position_embeddings = self.position_embeddings(triple_ids)

        embeddings = inputs_embeds + entity_embeddings + triple_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x .. #
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)

def get_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.eq(pad_idx).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(inputs[0].device)

    active_inputs = [collect_active_part(_, active_inst_idx, n_prev_active_inst, n_bm) for _ in inputs]

    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return active_inputs, active_inst_idx_to_position_map


def beam_decode_step(model, inst_dec_beams, len_dec_seq, active_inst_idx_list, inputs, \
                     inst_idx_to_position_map, n_bm, banwords):
    ''' Decode and update beam status, and then return active beam idx '''
    n_active_inst = len(inst_idx_to_position_map)

    # dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
    dec_partial_seq = [inst_dec_beams[idx].get_current_state()
                       for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
    dec_partial_seq = torch.stack(dec_partial_seq).to(inputs[0].device)
    dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)

    combined_inputs = inputs + [dec_partial_seq]
    #logits = model.forward_v2(*combined_inputs)[:, -1, :] / 1.
    src_lengths = inputs[0].shape[1]
    prev_output_tokens = dec_partial_seq
    logits = model.forward_v2(inputs, src_lengths, prev_output_tokens)
    logits = logits[:, -1, :] / 1

    if len_dec_seq <= 5:
        logits[:, banwords] = -np.inf

    word_prob = F.log_softmax(logits, dim=1)
    word_prob = word_prob.view(n_active_inst, n_bm, -1)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = []
    for inst_idx, inst_position in inst_idx_to_position_map.items():
        is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
        if not is_inst_complete:
            active_inst_idx_list += [inst_idx]

    return active_inst_idx_list


def collect_hypothesis_and_scores(pad_idx, inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for beam in inst_dec_beams:
        scores = beam.scores

        hyps = np.array([beam.get_hypothesis(i) for i in range(beam.size)], 'long')
        lengths = (hyps != pad_idx).sum(-1)
        normed_scores = [scores[i].item() / lengths[i] for i, hyp in enumerate(hyps)]
        idxs = np.argsort(normed_scores)[::-1]

        all_hyp.append([hyps[idx] for idx in idxs])
        all_scores.append([normed_scores[idx] for idx in idxs])

    return all_hyp, all_scores

class KgptEncoder(FairseqEncoder):

    def __init__(self, cfg: Kg2textModelConfig,  dictionary, embed_tokens): # embedding_layer cfg.model.kgpt
        super().__init__(None)
        """ original config and parameter for init encoder
        hidden_size = config.hidden_size, 
        d_inner, 
        n_head, 
        d_k, 
        d_v, 
        dropout=config.hidden_dropout_prob
        knowledge-config.json
        {
          "hidden_size": 768,
          "hidden_dropout_prob": 0.1,
          "max_entity_embeddings": 30,
          "max_triple_embeddings": 20,
          "max_position_embeddings": 1024,
          "layer_norm_eps": 1e-12,
          "sos_token_id": 50262,
          "eos_token_id": 50261,
          "vocab_size": 50264,
          "pad_token_id": 50263
        }
        argv.n_layers       
        """
        decoder_embed_dim = cfg.decoder_embed_dim
        cfg = cfg.kgpt
        hidden_size = cfg.encoder_embed_dim
        n_head = cfg.encoder_attention_heads
        n_layers = cfg.encoder_layers
        d_k = cfg.encoder_embed_dim // cfg.encoder_attention_heads
        d_v = cfg.encoder_embed_dim // cfg.encoder_attention_heads
        d_inner = cfg.encoder_ffn_embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.pad_idx = cfg.pad_token_id
        self.embedding = KnowledgeEmbeddings(cfg, embed_tokens)
        self.post_word_emb = PositionalEmbedding(d_model=hidden_size, max_len=1024)
        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, d_inner, n_head, d_k, d_v, dropout=cfg.encoder_layerdrop)
            for _ in range(n_layers)])
        if decoder_embed_dim != hidden_size:
            self.proj_to = nn.Linear(hidden_size, decoder_embed_dim)
            nn.init.normal_(self.proj_to.weight, mean=0, std=hidden_size ** -0.5)
        else:
            self.proj_to = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # original input: src_tokens=src_tokens, src_lengths=src_lengths
        # src_token = (input_ids, entity_ids, triple_ids, position_ids), src_lengths = args.max_enc_len 760 for wikidata
        enc_embed = self.embedding(src_tokens) # forward_embedding in transformer.py
        #encoder_embedding = enc_embed.deepcopy() # TODO: concern about the gradient
        x = enc_embed
        input_ids = src_tokens[0]
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)
        for layer in self.enc_layer_stack:
            x, _ = layer(x, non_pad_mask, slf_attn_mask)

        # x: B x len_src x C
        if self.proj_to:
            x = self.proj_to(x)

        encoder_padding_mask = input_ids.eq(self.padding_idx)

        return {
            "encoder_out": [x.transpose(0,1)],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T # eq(self.pad_idx)
            "encoder_embedding": [],  # B x T x C
            "encoder_states": None,  # List[T x B x C]
            "src_tokens": [input_ids], # B x T
            "src_lengths": [] # max_enc_len: 760 for pretrained kgpt model
        }

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        return self.embedding(src_tokens)

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

class KgptDecoder(FairseqIncrementalDecoder):

    def __init__(self, cfg: KgptConfig, dictionary, embed_token, output_projection=None): # orignial paras: d_inner, n_head, d_k, d_v, n_layers, dropout=0.1
        super().__init__(None) # dictionary?
        hidden_size = cfg.decoder_embed_dim
        n_head = cfg.decoder_attention_heads
        d_k = hidden_size // n_head
        d_v = hidden_size // n_head
        d_inner = hidden_size * 4
        n_layers = cfg.decoder_layers
        dropout = cfg.decoder_layerdrop  # where is this used for?
        self.padding_idx = cfg.pad_token_id
        self.sos_idx = cfg.sos_token_id
        self.eos_idx = cfg.eos_token_id
        self.d_inner = hidden_size * 4
        self.output_embed_dim = cfg.decoder_embed_dim

        self.positionwise_copy_prob = cfg.positionwise_copy_prob
        self.word_embeddings = embed_token
        self.post_word_emb = PositionalEmbedding(d_model=hidden_size, max_len=1024)
        self.dec_layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, self.d_inner, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)])
        if output_projection is None:
            self.gate = nn.Linear(hidden_size, 1)
            self.proj = nn.Linear(hidden_size, cfg.vocab_size)
            nn.init.normal_(self.proj.weight, mean=0, std=self.output_embed_dim ** -0.5)
            nn.init.normal_(self.gate.weight, mean=0, std=self.output_embed_dim ** -0.5)
        else:
            self.output_projection = output_projection
        self.adaptive_softmax = None


    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, features_only=False, **kwargs):
        """
        arguments needed:
        tgt_seq, (-> embedding(same as encode wordembedding) + post_word_emb-> dec_output)
        self.pad_idx(tgt_token),
        input_ids,
        self.pad_idx(src_token),
        enc_output
        """
        """
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

        # return dec_output, {
        #             "prev_output_tokens": prev_output_tokens,
        #             "encoder_out": encoder_out
        #         }

        """
        if features_only:
            return x
        # TODO: use dictionary to join net_output and logits
        else:
            full_vocab_prob = self.copy_gate_prob(prev_output_tokens, x, encoder_out, self.positionwise_copy_prob)
            logits = self.get_log_prob(full_vocab_prob)
            return x, {
                "logits": [logits]
            }
        """

    def extract_features(self, prev_output_tokens, encoder_out=None, tgt_seq=None, incremental_state=None):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # prev_output_tokens: output_ids[:-1]
        tgt_seq = prev_output_tokens
        input_ids, enc_output = encoder_out["src_tokens"][0], encoder_out["encoder_out"][0].transpose(0, 1)  # [bsz, len_k, d_k]
        non_pad_mask = get_non_pad_mask(tgt_seq, self.padding_idx)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=prev_output_tokens,
                                                     seq_q=prev_output_tokens, pad_idx=self.padding_idx)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=prev_output_tokens,
                                                  pad_idx=self.padding_idx)

        dec_output = self.word_embeddings(prev_output_tokens) + self.post_word_emb(prev_output_tokens)
        # B x T x C
        extra = None
        for layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
        # end of extract features
        return dec_output, {
            "prev_output_tokens": prev_output_tokens,
            "encoder_out": encoder_out
        }

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features


    def output_layer_backup(self, features, extra, positionwise_copy_prob=False, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        dec_output, enc_output, dec_enc_attn_mask = features, extra["encoder_out"], \
                                                    extra["dec_enc_attn_mask"]
        input_ids = extra["input_ids"]

        copy_gate = torch.sigmoid(self.gate(dec_output))
        in_vocab_prob = torch.softmax(self.proj(dec_output), -1)
        scores = torch.bmm(dec_output, enc_output.transpose(2, 1))
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)
        oov_vocab_prob = torch.softmax(scores, -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob
        full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1),
                                                      oov_vocab_prob * copy_gate)

        return full_vocab_prob

    def get_log_prob_backup(self, full_vocab_prob):
            return torch.log(full_vocab_prob + 1e-8)

    def greedy_decode_backup(self, src_tokens, src_lengths, prepend_label, tgt_lengths, **kwargs):
        # -- Encode source
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        # prepare prev_output_tokens
        input_ids, enc_output = encoder_out["src_tokens"][0], encoder_out["encoder_out"][0]
        batch_size = input_ids.shape[0]
        prev_output_tokens = torch.LongTensor(batch_size, 1).fill_(prepend_label).to(input_ids.device)
        #x = x.transpose(0, 1) # TODO transpose/ sos_idx_tgt

        for step in range(tgt_lengths):
            # transformer: return x, {"attn": [attn], "inner_states": inner_states}
            dec_output, extra = self.decoder.extract_features(
                prev_output_tokens,  encoder_out)

            dec_output = dec_output[:, -1, :].unsqueeze(1)
            full_vocab_prob = self.copy_gate_prob(prev_output_tokens, dec_output, encoder_out)

            decoded = torch.argmax(full_vocab_prob, -1)
            prev_output_tokens = torch.cat([prev_output_tokens, decoded], -1)

        return prev_output_tokens[:, 1:]

    def copy_gate_prob_backup(self, prev_output_tokens, features, encoder_out, positionwise_copy_prob= False):
        """Project features to the vocabulary size."""

        enc_output = encoder_out["encoder_out"][0].transpose(1, 0) #B x T x C
        input_ids = encoder_out["src_tokens"][0]
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=prev_output_tokens,
                                                  pad_idx=self.decoder.padding_idx)


        copy_gate = torch.sigmoid(self.decoder.gate(features))
        in_vocab_prob = torch.softmax(self.decoder.proj(features), -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        scores = torch.bmm(features, enc_output.transpose(2, 1)) # B x C x T
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)
        # B x T x vocab_sz
        full_vocab_prob = full_vocab_prob.scatter_add(2,
                                                      input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1),
                                                      oov_vocab_prob * copy_gate)

        return full_vocab_prob

class KgTriple(FairseqEncoder):
    def __init__(self, cfg, src_dict, embed_tokens):  # embedding_layer
        super().__init__(None)
        pass

class CopyGate(nn.Module):
    def __init__(self, cfg, dictionary):
        super(CopyGate, self).__init__()
        hidden_size = cfg.decoder_embed_dim

        self.gate = nn.Linear(hidden_size, 1)
        if hasattr(cfg, "vocab_size"):
            self.proj = nn.Linear(hidden_size, cfg.vocab_size)
        else:
            self.proj = nn.Linear(hidden_size, len(dictionary))
        nn.init.normal_(self.proj.weight, mean=0, std=hidden_size ** -0.5)
        nn.init.normal_(self.gate.weight, mean=0, std=hidden_size ** -0.5)
        logger.info("copy gate and proj layers are created.")

    def forward(self, features):
        """Project features to the vocabulary size."""

        copy_gate = torch.sigmoid(self.gate(features))
        output_proj_vec = self.proj(features)
        in_vocab_prob = torch.softmax(output_proj_vec, -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        return {
                "features": features,
                "copy_gate": copy_gate,
                "full_vocab_prob": full_vocab_prob,
                "vocab_vector": output_proj_vec
        }

class Kg2textTransformerModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        encoder = None
        if cfg.encoder_type.lower() == "kgpt":
            encoder = KgptEncoder(cfg, src_dict, embed_tokens)
        elif cfg.encoder_type.lower() == 'kgtriple':
            encoder = KgTriple(cfg, src_dict, embed_tokens)
        elif cfg.encoder_type.lower() in ["mbart50", "mbart50t"]:
            encoder = TransformerEncoder(cfg.mbart50, src_dict, embed_tokens)

        if getattr(cfg, "load_pretrained_encoder", False):
            #cfg.pretrained_encoder_file = "kgpt_kgpt_encoder.pt"
            #cfg.pretrained_encoder_file = get_kg2text_abs_path("encoder_load", cfg, cfg.pretrained_encoder_file)
            reloaded_encoder = torch.load(cfg.pretrained_encoder_file, map_location=torch.device('cpu'))
            #para2kgpt_mbart(encoder, reloaded_encoder)
            encoder.load_state_dict(reloaded_encoder, strict=True)
            #save_component("encoder", encoder, cfg)
            #encoder = checkpoint_utils.load_pretrained_component_from_model( # it requires the chpt is a FairseqEncoder
            #    component=encoder, checkpoint=cfg.pretrained_encoder_file
            #) essentially it is torch.load() then nn.module.load_state_dict
            logger.info(
                f"loaded pretrained encoder from: "
                f"{cfg.pretrained_encoder_file}"
            )

        return encoder

    @classmethod
    def build_decoder(cls, cfg: Kg2textModelConfig, tgt_dict, embed_tokens):

        decoder = None
        output_projection = cls.build_output_projection(cfg, tgt_dict, embed_tokens)
        if cfg.decoder_type.lower() == "kgpt":
            decoder = KgptDecoder(cfg.kgpt, tgt_dict, embed_tokens, output_projection)
        elif "mbart50" in cfg.decoder_type.lower():
            if output_projection:
                decoder = TransformerDecoder(cfg.mbart50, tgt_dict, embed_tokens, output_projection=output_projection) 
                #TODO: arg -> cfg copy
                """
                def __init__(
                    self,
                    args,
                    dictionary,
                    embed_tokens,
                    no_encoder_attn=False,
                    output_projection=None,
                ):           
                """
            else:
                decoder = TransformerDecoderScriptable(cfg.mbart50, tgt_dict, embed_tokens)  # TODO: Scriptable or Decoder

        if getattr(cfg, "load_pretrained_decoder", False):
            #cfg.pretrained_decoder_file = get_kg2text_abs_path("decoder_load", cfg, cfg.pretrained_decoder_file)
            reloaded_decoder = torch.load(cfg.pretrained_decoder_file, map_location=torch.device('cpu'))
            #printMyModel(reloaded_decoder)
            decoder.load_state_dict(reloaded_decoder, strict=False)
            """
            if cfg.decoder_type == "kgpt":
                dict_new = paraFromKgpt3(decoder, reloaded_decoder)
                decoder.load_state_dict(dict_new, strict=True)
            """
            #paraFromKgpt2(decoder, reloaded_decoder)
            #torch.save(reloaded_decoder.state_dict(), "mbart50_decoder.pt")
            #save_my_cfg(cfg, "config/mbart50_decoder.yaml")
            #paraFromKgpt(decoder, reloaded_decoder)


            #decoder = checkpoint_utils.load_pretrained_component_from_model(
                # it requires the chpt is a FairseqDecoder
            #    component=decoder, checkpoint=cfg.pretrained_decoder_file
            #)
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.pretrained_decoder_file}"
            )

        return decoder

    @classmethod
    def build_output_projection(cls, cfg: Kg2textModelConfig, dictionary, embed_tokens):
        if cfg.use_copy_gate:
            if cfg.decoder_type == "kgpt":
                output_projection = CopyGate(cfg.kgpt, dictionary)
            elif "mbart50" in cfg.decoder_type:
                output_projection = CopyGate(cfg.mbart50, dictionary)
            else:
                output_projection = None
                logger.info("no copy gate is implemented")
        else:
            output_projection = nn.Linear(
            cfg.decoder_embed_dim, len(dictionary), bias=False
        )
            nn.init.normal_(
                output_projection.weight, mean=0, std=cfg.decoder_embed_dim ** -0.5
            )
        return output_projection


    @classmethod
    def build_model(cls, cfg: Kg2textModelConfig, task=None):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder_embed_dim != cfg.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder_embed_file and (
                    cfg.decoder_embed_file != cfg.encoder_embed_file
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder_embed_dim, cfg.encoder_embed_file
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder_embed_dim, cfg.encoder_embed_file
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder_embed_dim, cfg.decoder_embed_file
            )

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        # decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
        #get_pretrained_components(cfg, encoder, decoder)
        #print("checked")
        return cls(encoder, decoder)

    @classmethod
    def build_decoder_embedding(cls, cfg: Kg2textConfig, task, tgt_dict):

        # tgt_dict = task.target_dictionary
        decoder_embed_tokens = cls.build_embedding(
            cfg, tgt_dict, cfg.model.decoder_embed_dim, cfg.model.decoder_embed_file
        )
        return decoder_embed_tokens

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        if "mbart" in cfg.decoder_type and num_embeddings > 60000:
            padding_idx = dictionary.pad()
        else:
            padding_idx = cfg.kgpt.pad_token_id
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        # TODO: load embedding layer?
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, src_tokens, src_lengths, prev_output_tokens,**kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out, extra = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
        )
        # mbart50 decoder: return x, {"attn": [attn], "inner_states": inner_states}
        # kgpt decoder: return x, {logits: [logits]}
        return decoder_out, extra

    def greedy_decode(self, src_tokens, src_lengths, prepend_label, tgt_lengths, **kwargs):
        # -- Encode source
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        # prepare prev_output_tokens
        input_ids, enc_output = encoder_out["src_tokens"][0], encoder_out["encoder_out"][0]
        batch_size = input_ids.shape[0]
        prev_output_tokens = torch.LongTensor(batch_size, 1).fill_(prepend_label).to(input_ids.device)
        #x = x.transpose(0, 1) # TODO transpose/ sos_idx_tgt

        for step in range(tgt_lengths):
            # transformer: return x, {"attn": [attn], "inner_states": inner_states}
            dec_output, extra = self.decoder.extract_features(
                prev_output_tokens,  encoder_out)
            # kgpt: return dec_output, {
            #                      "prev_output_tokens": prev_output_tokens,
            #                      "encoder_out": encoder_out
            #                  }
            # mbart50: return x, {"attn": [attn], "inner_states": inner_states}

            dec_output = dec_output[:, -1, :].unsqueeze(1)
            full_vocab_prob = self.copy_gate_prob(prev_output_tokens, dec_output, encoder_out)
            # TODO mbart50 has no copy_gate_prob()
            decoded = torch.argmax(full_vocab_prob, -1)
            prev_output_tokens = torch.cat([prev_output_tokens, decoded], -1)

        return prev_output_tokens[:, 1:]

    def copy_gate_prob(self, prev_output_tokens, features, encoder_out, log_prob = True):
        """Project features to the vocabulary size."""

        enc_output = encoder_out["encoder_out"][0].transpose(1, 0) #B x T x C
        input_ids = encoder_out["src_tokens"][0]
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=prev_output_tokens,
                                                  pad_idx=self.decoder.padding_idx)


        copy_gate = torch.sigmoid(self.decoder.output_projection.gate(features))
        in_vocab_prob = torch.softmax(self.decoder.output_projection.proj(features), -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        scores = torch.bmm(features, enc_output.transpose(2, 1)) # B x C x T
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)
        # B x T x vocab_sz
        full_vocab_prob = full_vocab_prob.scatter_add(2,
                                                      input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1),
                                                      oov_vocab_prob * copy_gate)

        return full_vocab_prob

    def copy_gate_prob_v2(self, prev_output_tokens, features, encoder_out, copy_gate, full_vocab_prob,
                          positionwise_copy_prob = False, prob=True):
        """Project features to the vocabulary size."""

        enc_output = encoder_out["encoder_out"][0].transpose(1, 0)  # B x T x C
        input_ids = encoder_out["src_tokens"][0]
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=prev_output_tokens,
                                                  pad_idx=self.decoder.padding_idx)

        #copy_gate = torch.sigmoid(self.decoder.gate(features))
        #in_vocab_prob = torch.softmax(self.decoder.proj(features), -1)
        #full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        scores = torch.bmm(features, enc_output.transpose(2, 1))  # B x C x T
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)
        # B x T x vocab_sz
        full_vocab_prob = full_vocab_prob.scatter_add(2,
                                                      input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1),
                                                      oov_vocab_prob * copy_gate)

        if positionwise_copy_prob:
            return torch.log(full_vocab_prob + 1e-8), oov_vocab_prob * copy_gate
        else:
            return torch.log(full_vocab_prob + 1e-8)

    def forward_v2(self, src_tokens, src_lengths, prev_output_tokens,**kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out, extra = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
        )
        # mbart50 decoder: return x, {"attn": [attn], "inner_states": inner_states}
        # kgpt decoder: return x, {logits: [logits]}
        # copy_gate_ output_projection:
        """
        {   
            "features": features
            "copy_gate": copy_gate,
            "full_vocab_prob": full_vocab_prob,
            "vocab_vector": output_proj_vec
        }
        """
        features, copy_gate, full_vocab_prob = decoder_out["features"], decoder_out["copy_gate"], decoder_out["full_vocab_prob"]
        logits = self.copy_gate_prob_v2(prev_output_tokens, features, encoder_out, copy_gate, full_vocab_prob,
                          positionwise_copy_prob = False, prob=True)
        return logits

    def get_log_probs(self, net_output, log_probs = True, sample: Optional[Dict[str, Tensor]] = None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        in_vocab_prob = torch.softmax(net_output, -1)
        lprobs = torch.log(in_vocab_prob + 1e-8)

        return lprobs

    def beam_search(self, inputs, tokenizer, n_bm, max_token_seq_len=30, banwords=[]):

        with torch.no_grad():
            # -- Repeat data for beam search
            n_inst, len_s = inputs[0].size()  # len_s: 256, n_inst: 16 (B: batch_size) inputs: 4 x B x len
            # inputs: 4 x B x len / x: B x len -> B x (len x n_bm)  -> (B x n_bm) x len
            inputs = [x.repeat(1, n_bm).view(n_inst * n_bm, len_s) for x in inputs]

            # -- Prepare beams /one beam per sample in the batch
            inst_dec_beams = [Beam(n_bm, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id,
                                   device=inputs[0].device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # return {inst_idx: tensor_position,....,} for inst_idx is currently active

            # -- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(self, inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        inputs, inst_idx_to_position_map, n_bm, banwords)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs, inst_idx_to_position_map = collate_active_info(inputs, inst_idx_to_position_map,
                                                                       active_inst_idx_list, n_bm)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(tokenizer.pad_token_id, inst_dec_beams, n_bm)

            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])

            return result


def get_pretrained_components(model_cfg: Kg2textModelConfig, encoder=None, decoder=None):
    
    if (model_cfg.encoder_type, model_cfg.decoder_type) == ("kgpt", "kgpt"):
        label = "fairseq_kgpt_kgpt"
    elif (model_cfg.encoder_type, model_cfg.decoder_type) == ("kgpt", "mbart50"):
        label = "fairseq_kgpt_mbart50_encoder"
    elif (model_cfg.encoder_type, model_cfg.decoder_type) == ("mbart50", "mbart50"):
        label = "fairseq_mbart50"
    elif (model_cfg.encoder_type, model_cfg.decoder_type) == ("mbart50t", "mbart50t"): # TODO
        label = "fairseq_mbart50_with_tags"


    if label == "fairseq_kgpt_kgpt":
        # kgpt -> fairseq_kgpt_kgpt_encoder, fairseq_kgpt_kgpt_decoder
        # model_new.load_state_dict(dict_new)
        # with module prefix
        kgpt_state_dict = load_kgpt_state_dict(model_cfg) 
        para_copy_kgpt_kgpt_encoder(encoder, kgpt_state_dict)
        paraFromKgpt3(decoder, kgpt_state_dict)

        save_encoder_file = "kgpt_kgpt_encoder.pt"
        save_decoder_file = "kgpt_kgpt_decoder.pt"
        abs_save_encoder_file = get_kg2text_abs_path("encoder_save", model_cfg, save_encoder_file)
        abs_save_decoder_file = get_kg2text_abs_path("decoder_save", model_cfg, save_decoder_file)

        #torch.save(encoder.state_dict(), abs_save_encoder_file)
        torch.save(decoder.state_dict(), abs_save_decoder_file)
        
    elif label == "fairseq_kgpt_mbart50_encoder":
        # kgpt -> kgpt_mbart50_encoder
        kgpt_state_dict = load_kgpt_state_dict(model_cfg)
        para_copy_kgpt_kgpt_encoder(encoder, kgpt_state_dict)

        save_encoder_file = "kgpt_mbart50_encoder.pt"
        abs_save_encoder_file = get_kg2text_abs_path("encoder_save", model_cfg, save_encoder_file)
        torch.save(encoder.state_dict(), abs_save_encoder_file)
    
    elif label == "fairseq_mbart50":
        # mbart50 -> kgpt_mbart50_decoder with copy_gate w/o tags
        # mbart50 -> mbart50_mbart50_encdoer
        mbart50_model, cfg_x, task_x = load_mbart50_whole(model_cfg)
        decoder_state_dict = mbart50_model.decoder.state_dict()
        if model_cfg.use_copy_gate:
            decoder.load_state_dict(decoder_state_dict, strict = False)
        else:
            decoder.load_state_dict(decoder_state_dict, strict = True)

        save_checkpoint_file = "mbart50.pt"
        save_encoder_file = "mbart50_mbart50_encoder_wotags.pt"
        save_decoder_file = "mbart50_mbart50_decoder_wotags.pt"
        abs_save_checkpoint_file = get_kg2text_abs_path("model_mbart50", model_cfg, save_checkpoint_file)
        abs_save_encoder_file = get_kg2text_abs_path("encoder_save", model_cfg, save_encoder_file)
        abs_save_decoder_file = get_kg2text_abs_path("decoder_save", model_cfg, save_decoder_file)

        torch.save(mbart50_model.state_dict(), abs_save_checkpoint_file) # TODO add to modelconfig
        torch.save(decoder.state_dict(), abs_save_decoder_file)
        torch.save(mbart50_model.encoder.state_dict(), abs_save_encoder_file)

    elif label == "fairseq_mbart50_with_tags":
        # mbart50_decoder -> mbart50_mbart50_decoder_with_tag
        # mbart50_encoder -> mbart50_mbart50_encoder_with_tag
        mbart50_model, cfg_x, task_x = load_mbart50_whole(model_cfg)
        decoder_state_dict = mbart50_model.decoder.state_dict()
        encoder_state_dict = mbart50_model.encoder.state_dict()

        para_copy_mbart50t_mbart50t(encoder, encoder_state_dict)
        para_copy_mbart50t_mbart50t(decoder, decoder_state_dict)

        save_encoder_file = "mbart50_mbart50_encoder_wtags.pt"
        save_decoder_file = "mbart50_mbart50_decoder_wtags.pt"
        abs_save_encoder_file = get_kg2text_abs_path("encoder_save", model_cfg, save_encoder_file)
        abs_save_decoder_file = get_kg2text_abs_path("decoder_save", model_cfg, save_decoder_file)
        torch.save(decoder.state_dict(), abs_save_encoder_file)
        torch.save(encoder.state_dict(), abs_save_decoder_file)

    else:
        raise NotImplementedError

