# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
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
from fairseq.models.transformer import TransformerModel, FairseqEncoderDecoderModel, FairseqEncoder, \
    FairseqIncrementalDecoder, Embedding, TransformerDecoder
from fairseq.models.bart.model import BARTModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import checkpoint_utils, utils
from fairseq.models.bart.hub_interface import BARTHubInterface

from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable
import pickle
from fairseq.models.bart import BARTModel
from fairseq.data import encoders
import os
import Model
from myutils import *
from torch.nn import Parameter
from kg2textConfig import *

logger = logging.getLogger(__name__)



def save_cfg(loaded):
    with open("config/loaded", "wb") as f:
        pickle.dump(loaded, f)
        f.flush()
        assert f.seek(0) == 0

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
    def __init__(self, config, word_emb): # config = knowledge.json
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
            self.word_embeddings = nn.Embedding(config.vocab_size_src, config.hidden_size_enc, padding_idx=config.pad_token_id_src)
        self.entity_embeddings = nn.Embedding(config.max_entity_embeddings, config.hidden_size_enc)
        self.triple_embeddings = nn.Embedding(config.max_triple_embeddings, config.hidden_size_enc)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size_enc)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size_enc, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob_enc)

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

class KgptEncoder(FairseqEncoder):

    def __init__(self, cfg, embed_tokens): # embedding_layer cfg.model.kgpt
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
          "vocab_size": 50264,
          "pad_token_id": 50263,
          "hidden_dropout_prob": 0.1,
          "max_entity_embeddings": 30,
          "max_triple_embeddings": 20,
          "max_position_embeddings": 1024,
          "layer_norm_eps": 1e-12,
          "sos_token_id": 50262,
          "eos_token_id": 50261,
          "hidden_size": 768
        }
        argv.n_layers       
        """
        hidden_size = cfg.hidden_size_enc
        n_head = cfg.n_heads_enc
        n_layers = cfg.n_layers_enc
        d_k = cfg.hidden_size_enc // n_head
        d_v = cfg.hidden_size_enc // n_head
        d_inner = cfg.hidden_size_enc * 4

        self.pad_idx = cfg.pad_token_id_src
        self.embedding = KnowledgeEmbeddings(cfg, embed_tokens)
        self.post_word_emb = PositionalEmbedding(d_model=hidden_size, max_len=1024)
        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, d_inner, n_head, d_k, d_v, dropout=cfg.hidden_dropout_prob_enc)
            for _ in range(n_layers)])
        if getattr(cfg, "hidden_size_dec", hidden_size) != hidden_size:
            self.proj = nn.Linear(hidden_size, cfg.hidden_size_dec)
        else:
            self.proj = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # original input: src_tokens=src_tokens, src_lengths=src_lengths
        # src_token = (input_ids, entity_ids, triple_ids, position_ids), src_lengths = argv.max_enc_len
        enc_embed = self.embedding(src_tokens)
        x = enc_embed.deepcopy() # TODO: concern about the gradient
        input_ids = src_tokens[0]
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)
        for layer in self.enc_layer_stack:
            x, _ = layer(x, non_pad_mask, slf_attn_mask)
        x.transpose(0, 1)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [non_pad_mask],  # B x T
            "encoder_embedding": [enc_embed],  # B x T x C
            "encoder_states": None,  # List[T x B x C]
            "src_tokens": [input_ids],
            "src_lengths": [],
        }

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

    def __init__(self, cfg, dictionary, embed_token): # orignial paras: d_inner, n_head, d_k, d_v, n_layers, dropout=0.1
        super().__init__(None) # dictionary?
        hidden_size = cfg.hidden_size_dec
        n_head = cfg.n_heads_dec
        d_k = cfg.hidden_size_dec // n_head
        d_v = cfg.hidden_size_dec // n_head
        d_inner = cfg.hidden_size_dec * 4
        n_layers = cfg.n_layers_dec
        dropout = cfg.hidden_dropout_prob_dec  # where is this used for?
        self.pad_idx_src = cfg.pad_token_id_src
        self.pad_idx_tgt = cfg.pad_token_id_tgt
        self.sos_idx = cfg.sos_token_id_tgt
        self.eos_idx = cfg.eos_token_id_tgt
        self.d_inner = cfg.hidden_size_dec*4


        self.positionwise_copy_prob = cfg.positionwise_copy_prob
        self.word_embeddings = embed_token
        self.post_word_emb = PositionalEmbedding(d_model=hidden_size, max_len=1024)
        self.dec_layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, self.d_inner, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)])
        self.gate = nn.Linear(cfg.hidden_size_dec, 1)
        self.proj = nn.Linear(cfg.hidden_size_dec, cfg.vocab_size_tgt)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
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
        # tgt_seq, input_ids, is_net_output
        tgt_seq, is_net_output = None, False
        for key, val in kwargs.items():
            if key == "target_seq":
                tgt_seq = val
            if key == "is_net_output":
                is_net_output = val

        x, extra = self.extract_features(
            prev_output_tokens, encoder_out)
        if is_net_output:
            net_output = self.output_layer(x)
        # TODO: use dictionary to join net_output and logits
        else:
            return self.get_log_probs(x, encoder_out, extra["dec_enc_mask"],
                     self.positionwise_copy_prob)

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
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx_tgt)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx_tgt)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx_src)

        dec_output = self.embedding(prev_output_tokens) + self.post_word_emb(prev_output_tokens)
        enc_output = encoder_out["encoder_out"][0].transpose(0, 1)  # [bsz, len_k, d_k]

        for layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        return dec_output, {
            "dec_enc_mask": dec_enc_attn_mask  # [bsz, len_dec, len_enc]
            }

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        in_vocab_prob = torch.softmax(self.proj(features), -1)
        return in_vocab_prob

    def get_log_probs(self, features, encoder_out=None, dec_enc_mask=None, input_ids=None,
                     positionwise_copy_prob=False):

        input_ids, enc_output = encoder_out["src_tokens"][0], encoder_out["encoder_out"][0]
        in_vocab_prob = torch.softmax(self.proj(features), -1)
        copy_gate = torch.sigmoid(self.gate(features))
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob


        scores = torch.bmm(features, encoder_out.transpose(2, 1))
        scores = scores.masked_fill(dec_enc_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)
        full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1),
                                                      oov_vocab_prob * copy_gate)

        if positionwise_copy_prob:
            return torch.log(full_vocab_prob + 1e-8), oov_vocab_prob * copy_gate
        else:
            return torch.log(full_vocab_prob + 1e-8)

class KgTriple(FairseqEncoder):
    def __init__(self, cfg, embed_tokens):  # embedding_layer
        super().__init__(None)
        pass

class Kg2textTransformerModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, emb_layer):
        encoder = KgptEncoder(cfg, emb_layer)
        if cfg.encoder_type.lower() == "kgpt":
            encoder = KgptEncoder(cfg, emb_layer)
            printMyModel(encoder)


            print(torch.cuda.is_available())
            # kgpt encoder
            #model_trained = Model0.GatedTransformerDecoder(config, 8, 6) # config.json
            # reloaded = torch.load(model_cfg.load_from,)
            reloaded = torch.load("/media/MyDataStor1/jxian/efs-storage/Kg2text/model/kgpt/model_ep14.pt", map_location=torch.device('cpu'))
            #new_encoder = paraFromKgpt(encoder, reloaded)

            dict_new = encoder.state_dict()
            dict_trained = reloaded
            list_new = list(dict_new.keys())
            list_trained = list(dict_trained.keys())
            L = len(list_new) - 2
            j = 0
            for i in range(L):
                name = list_trained[i]
                new_name = list_new[i]
                param = dict_trained[name]
                new_param = dict_new[new_name]
                if param.shape != new_param.shape:
                    print(name, new_name)

                if 'entity_embeddings' in name or 'triple_embeddings' in name:
                    if param.shape != new_param.shape:
                        print("Reinitializing the weight for {}".format(name))
                        continue
                if isinstance(param, Parameter):
                    param = param.data
                dict_new[new_name].copy_(param)
            print(L)
            torch.save(encoder.state_dict(), "/media/MyDataStor1/jxian/efs-storage/Kg2text/model/kgpt_encoder_dict.pt")
            torch.save(encoder, "/media/MyDataStor1/jxian/efs-storage/Kg2text/model/kgpt_encoder.pt")


            if cfg.load_pretrained_encoder_from:
                encoder = checkpoint_utils.load_pretrained_component_from_model( # it requires the chpt is a FairseqEncoder
                    component=encoder, checkpoint=cfg.load_pretrained_encoder_from
                )# TODO: we still need a module to split kgpt into fairseq's encoder and decoder
                logger.info(
                    f"loaded pretrained encoder from: "
                    f"{cfg.load_pretrained_encoder_from}"
                )
        elif cfg.encoder_type.lower() == 'kgtriple':
            encoder = KgTriple(cfg, emb_layer)
        return encoder

    @classmethod
    def build_decoder(cls, cfg: Kg2textConfig, tgt_dict, embed_tokens): # Kg2textConfig
        cfg_mbart50 = TransformerConfig()
        mycfg_mbart = cfg.mbart50
        dir1 = mycfg_mbart.__dir__()
        for key in dir1:
            val = mycfg_mbart[key]
            setattr(cfg_mbart50, key, val)
        decoder = TransformerDecoderScriptable(cfg_mbart50, tgt_dict, embed_tokens) # TODO: rewrite cfg?
        if cfg.decoder_type.lower() == "kgpt":  # TODO: need to assert cfg == loaded model?
            decoder = KgptDecoder(cfg, tgt_dict, embed_tokens)
            if getattr(cfg, "load_pretrained_decoder_from", None):
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=cfg.load_pretrained_decoder_from
                )
                logger.info(
                    f"loaded pretrained decoder from: "
                    f"{cfg.load_pretrained_decoder_from}"
                )

        if "mbart" in cfg.decoder_type.lower():
            if cfg.load_pretrained_decoder_from:
            # decoder = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from).model.decoder
                cp = '/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50/model.pt'
                mbart, cfg_mbart, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp], arg_overrides={
                    'lang_dict': '/media/MyDataStor1/jxian/efs-storage/Kg2text/tokenizer/mbart50/dict/dict.txt', # ML50_langs.txt
                    'data': '/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50'})
                decoder = mbart[0].decoder
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.load_pretrained_decoder_from}"
            )
        return decoder

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

        #decoder_embed_tokens = cls.build_decoder_embedding(cfg, None, tgt_dict)
        if cfg.encoder_type == "kgpt" == cfg.decoder_type:
            # kgpt enc dec share word embedding layer
            encoder = cls.build_encoder(cfg, decoder_embed_tokens)
        else:
            encoder = cls.build_encoder(cfg, None)
        # TODO: get parameters from pretrained kgpt model
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        # decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
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
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        # TODO: load embedding layer?
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
        )
        return decoder_out







