from transformers import GPT2Tokenizer
from dataclasses import MISSING
from logging import Logger, LoggerAdapter
import torch
import argparse
import os
from omegaconf import OmegaConf
from torch.nn import Parameter
#from fairseq import checkpoint_utils, utils
#from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from fairseq.data import Dictionary
from kg2textConfig import *
from omegaconf.dictconfig import DictConfig
from fairseq import checkpoint_utils, utils
from collections import OrderedDict
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


import torch
import torch.nn as nn
import numpy as np



def myparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, default="",
                        help="maximum length of the decoding part")
    parser.add_argument("--option", type=str, default="eval", help="select option for tasks: train/finetune/eval/test" )
    parser.add_argument("--data", type=str, default="webnlg", help="select dataset for task: wikidata/webnlg/wikibio")
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoint", help="checkpoint foler path relative for project dir")
    parser.add_argument("--dict_add_tags_src", type=bool, default=False, help="add tags symbols to source dictionary?")
    parser.add_argument("--dict_add_tags_tgt", type=bool, default=False, help="add tags symbols to target dictionary?")
    args = parser.parse_args()

    return args



class myMbart50Tokenizer:
    def __init__(self, token_cfg, label="tgt"):
        if label == "tgt":
            self.token_dict = Dictionary.load(token_cfg.tgt_vocab_filename)
            self.bpe_tokenizer = SentencepieceBPE(token_cfg.tgt_bpe)
            # path tgt_bpe.sentencepiece_model is a path
        elif label == "src":
            self.token_dict = Dictionary.load(token_cfg.src_vocab_filename)
            self.bpe_tokenizer = SentencepieceBPE(token_cfg.src_bpe)

    def encode(self, text):
        tokenized = self.bpe_tokenizer.encode(text)
        text_done = self.token_dict.encode_line(
            tokenized, add_if_not_exist=False, append_eos=True
        ).long()
        return text_done

    def decode(self, ids):
        tokenized = [ self.token_dict[i] for i in ids ]
        subwords = " ".join(tokenized)
        text_done = self.bpe_tokenizer.decode(subwords)
        return text_done

def get_project_dir(project="Kg2text"):
    path_cwd = os.path.abspath(os.getcwd())
    assert project in path_cwd
    path_cwd_list = path_cwd.split(os.sep)
    idx = path_cwd_list.index(project)
    project_path = os.sep.join(path_cwd_list[:idx + 1])
    return project_path

def get_my_args():
    parser = argparse.ArgumentParser()
    project_path = get_project_dir()
    config_path = os.path.join(project_path, "code", "kg2text_config.yaml")
    parser.add_argument("--data", type=str, default="test", help="specify webnlg/wikidata/wikibionlg")
    parser.add_argument('--option', type=str, default="eval", help="option: train/finetune/eval/test")
    parser.add_argument("--config", type=str, default=config_path, help="specify config file")
    parser.add_argument("--load_from", type=str, default="", help="specify model file")
    # TODO  remove or add to cfg?
    parser.add_argument("--dict_add_tags_src", type=bool, default=False, help="add tags symbols to source dictionary?")
    parser.add_argument("--dict_add_tags_tgt", type=bool, default=False, help="add tags symbols to target dictionary?")
    args = parser.parse_args()
    return args

def get_abs_path0(file_name, **kwargs):
    # model, tokenizer, config,
    father_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    grandfather_path = os.path.abspath(os.path.join(father_path, ".."))
    abs_path = os.path.join(father_path, file_name)
    # model.pt -> path/to/project/model/model_name/model.pt
    # dict.txt -> path/to/project/tokenizer/dict/dict.txt
    # model.bpe -> path/to/project/tokenizer/bpe/model.bpe

    # config.yaml -> path/to/project/config/config.yaml
    # data/task.json -> path/to/project/dataset/datasetname/task.json
    if "code" not in os.getcwd():
        # just add project path
        pass
    elif "code" in os.getcwd():
        # go to father
        pass
    return abs_path

def get_kg2text_abs_path(label, cfg, path, project="Kg2text", project_path = None):

    # get file name
    os_sep = os.sep
    path_sep = path.split(os.sep)
    if len(path_sep) == 1:
        file_name = path_sep
    else:
        file_name = path_sep[-1]
    
    Logger.info("detect file name: %s", file_name)

    # get project_path
    if project_path is None:
        path_cwd = os.path.abspath(os.getcwd())
        assert project in path_cwd
        path_cwd_list = path_cwd.split(os.sep)
        idx = path_cwd_list.index(project)
        project_path = os_sep.join(path_cwd_list[:idx + 1])
    

    abs_file_path = MISSING
    # construct abs path
    if label == "dict":
        assert isinstance(cfg, Kg2textDataSetConfig)
        abs_file_path = os.path.join(project_path, "dataset", "dict", path)

    elif label == "checkpoint":
        assert isinstance(cfg, Kg2textCommonConfig)
        abs_file_path = os.path.join(project_path, "checkpoint", path)

    elif label == "encoder_load":
        assert isinstance(cfg, Kg2textModelConfig)
        abs_file_path = os.path.join(project_path, "model", "encoder", path)

    elif label == "decoder_load":
        assert isinstance(cfg, Kg2textModelConfig)
        abs_file_path = os.path.join(project_path, "model", "decoder", path)

    elif label == "encoder_save":
        assert isinstance(cfg, Kg2textModelConfig)
        abs_file_path = os.path.join(project_path, "model", "encoder", "save", path)

    elif label == "decoder_save":
        assert isinstance(cfg, Kg2textModelConfig)
        abs_file_path = os.path.join(project_path, "model", "decoder", "save", path)

    elif "model_kgpt" in label:
        assert isinstance(cfg, Kg2textModelConfig)
        _, model_type = label.split("_")
        abs_file_path = os.path.join(project_path, "model", model_type, path)

    elif "model_mbart50" in label:
        assert isinstance(cfg, Kg2textModelConfig)
        _, model_type = label.split("_")
        abs_file_path = os.path.join(project_path, "model", model_type, path)
    elif label == "bpe":
        if isinstance(cfg, SentencepieceConfig):
            abs_file_path = os.path.join(project_path, "tokenizer", "mbart50", "bpe", path)
        elif isinstance(cfg, GPT2BPEConfig):
            abs_file_path = os.path.join(project_path, "tokenizer", "gpt2", "bpe", path)
        elif isinstance(cfg, KgptGpt2Config):
            abs_file_path = os.path.join(project_path, path)
    elif label == "data":
        assert isinstance(cfg, Kg2textDataSetConfig)
        abs_file_path = os.path.join(project_path, "dataset", cfg.dataset, path)
    elif label == "knowledge_file":
        assert isinstance(cfg, KgptConfig)
        abs_file_path = os.path.join(project_path, "preprocess", path)

    # TODO: check path ending type and using logger
    logger.info("loaded/saved {label} from {path}".format(label=label, path=abs_file_path))
    assert abs_file_path != MISSING

    return abs_file_path

def abspath_data_cfg(data_cfg):
    data_cfg.train_file = get_kg2text_abs_path("data", data_cfg, data_cfg.train_file)
    data_cfg.eval_file = get_kg2text_abs_path("data", data_cfg, data_cfg.eval_file)
    data_cfg.test_file = get_kg2text_abs_path("data", data_cfg, data_cfg.test_file)
    data_cfg.vocab_file_src = get_kg2text_abs_path("dict", data_cfg, data_cfg.vocab_file_src)
    data_cfg.vocab_file_tgt = get_kg2text_abs_path("dict", data_cfg, data_cfg.vocab_file_tgt)

def abspath_model_cfg(cfg, model_cfg, data_cfg):
    # absolute path
    # model
    # model.kgpt
    model_cfg.pretrained_encoder_file = get_kg2text_abs_path("encoder_load", model_cfg,
                                                             model_cfg.pretrained_encoder_file)
    model_cfg.pretrained_decoder_file = get_kg2text_abs_path("decoder_load", model_cfg,
                                                             model_cfg.pretrained_decoder_file)
    model_cfg.save_encoder_file = get_kg2text_abs_path("encoder_load", model_cfg, model_cfg.save_encoder_file)
    model_cfg.save_decoder_file = get_kg2text_abs_path("decoder_load", model_cfg, model_cfg.save_decoder_file)

    # path reset for dataset
    # current_path = os.path.abspath(__file__)

    if isinstance(model_cfg, KgptConfig):
        model_cfg.knowledge_file = get_kg2text_abs_path("knowledge_file", cfg, model_cfg.knowledge_file)
        data_cfg.knowledge_file = model_cfg.knowledge_file

def check_cfg(cfg):
    task_cfg, data_cfg, model_cfg, common, token_cfg = cfg.task, getattr(cfg.task,
                                                                         cfg.task.dataset), cfg.model, cfg.common, cfg.tokenizer
    
    # check consistency between task_cfg and data_cfg
    if task_cfg.encoder_type == "kgpt":
        assert data_cfg.tokenizer_src in ["kgpt_gpt2", "fairseq_gpt2"]
        assert "gpt" in data_cfg.vocab_file_src
    elif task_cfg.encoder_type in ["mbart", "mbart50"]:
        assert data_cfg.tokenizer_src in ["mbart", "mbart50", "sentencepiece"]
        assert "mbart" in data_cfg.vocab_file_src
    elif task_cfg.encoder_type in ["mbartt", "mbart50t"]:
        assert data_cfg.tokenizer_src in ["mbart", "mbart50", "sentencepiece"]
        assert "mbartt" in data_cfg.vocab_file_src or "mbart50t" in data_cfg.vocab_file_src

    if task_cfg.decoder_type == "kgpt":
        assert data_cfg.tokenizer_tgt in ["kgpt_gpt2", "fairseq_gpt2"]
        assert "gpt" in data_cfg.vocab_file_tgt
    elif task_cfg.decoder_type in ["mbart", "mbart50"]:
        assert data_cfg.tokenizer_tgt in ["mbart", "mbart50", "sentencepiece"]
        assert "mbart" in data_cfg.vocab_file_tgt
    elif task_cfg.decoder_type in ["mbartt", "mbart50t"]:
        assert data_cfg.tokenizer_tgt in ["mbart", "mbart50", "sentencepiece"]
        assert "mbartt" in data_cfg.vocab_file_tgt or "mbart50t" in data_cfg.vocab_file_tgt
    
    # token_cfg == data_cfg
    token_cfg.tokenizer_src = data_cfg.tokenizer_src
    token_cfg.tokenizer_tgt = data_cfg.tokenizer_tgt
    token_cfg.vocab_file_src = data_cfg.vocab_file_src
    token_cfg.vocab_file_tgt = data_cfg.vocab_file_tgt

    # model_cfg == task_cfg
    model_cfg.encoder_type = task_cfg.encoder_type
    model_cfg.decoder_type = task_cfg.decoder_type
    model_cfg.src_length = data_cfg.src_length
    model_cfg.tgt_length = data_cfg.tgt_length

    if model_cfg.encoder_type == "kgpt":
        model_cfg.encoder_embed_dim = model_cfg.kgpt.encoder_embed_dim
        if model_cfg.decoder_type == "kgpt":
            model_cfg.decoder_embed_dim = model_cfg.kgpt.decoder_embed_dim
            model_cfg.pretrained_encoder_file = "kgpt_kgpt_encoder.pt"
            model_cfg.pretrained_decoder_file = "kgpt_kgpt_decoder.pt"
            model_cfg.save_encoder_file = "kgpt_kgpt_encoder.pt"
            model_cfg.save_decoder_file = "kgpt_kgpt_decoder.pt"
        elif model_cfg.decoder_type == "mbart50":
            model_cfg.decoder_embed_dim = model_cfg.mbart50.decoder_embed_dim
            model_cfg.pretrained_encoder_file = "kgpt_mbart50_encoder.pt"
            model_cfg.pretrained_decoder_file = "kgpt_mbart50_decoder.pt"
            model_cfg.save_encoder_file = "kgpt_mbart50_encoder.pt"
            model_cfg.save_decoder_file = "kgpt_mbart50_decoder.pt"
        elif model_cfg.decoder_type == "mbart50t":
            model_cfg.decoder_embed_dim = model_cfg.mbart50.decoder_embed_dim
            model_cfg.pretrained_encoder_file = "kgpt_mbart50t_encoder.pt"
            model_cfg.pretrained_decoder_file = "kgpt_mbart50t_decoder.pt"
            model_cfg.save_encoder_file = "kgpt_mbart50t_encoder.pt"
            model_cfg.save_decoder_file = "kgpt_mbart50t_decoder.pt"

    elif model_cfg.encoder_type == "mbart50":
        model_cfg.encoder_embed_dim = model_cfg.mbart50.encoder_embed_dim
        if model_cfg.decoder_type == "mbart50":
            model_cfg.decoder_embed_dim = model_cfg.mbart50.decoder_embed_dim
            model_cfg.pretrained_encoder_file = "mbart50_mbart50_encoder.pt"
            model_cfg.pretrained_decoder_file = "mbart50_mbart50_decoder.pt"
            model_cfg.save_encoder_file = "mbart50_mbart50_encoder.pt"
            model_cfg.save_decoder_file = "mbart50_mbart50_decoder.pt"
    
    elif model_cfg.encoder_type == "mbart50t":
        model_cfg.encoder_embed_dim = model_cfg.mbart50.encoder_embed_dim
        if model_cfg.decoder_type == "mbart50t":
            model_cfg.decoder_embed_dim = model_cfg.mbart50.decoder_embed_dim
            model_cfg.pretrained_encoder_file = "mbart50_mbart50t_encoder.pt"
            model_cfg.pretrained_decoder_file = "mbart50_mbart50t_decoder.pt"
            model_cfg.save_encoder_file = "mbart50_mbart50t_encoder.pt"
            model_cfg.save_decoder_file = "mbart50_mbart50t_decoder.pt"
    if data_cfg.tokenizer_src != data_cfg.tokenizer_tgt:
        model_cfg.share_all_embeddings = False
            
def set_cfg(cfg_new, args):
    cfg_load = load_my_cfg(args.config)
    cfg_load.task.option, cfg_load.task.dataset = args.option, args.data
    cfg = myConfigUpdate(cfg_load, cfg_new)

    task_cfg, data_cfg, model_cfg = cfg.task, getattr(cfg.task, args.data), cfg.model
    # TODO update path?, save all the parameters
    
    # data_cfg = getattr(task_cfg, task_cfg.dataset)
    data_cfg.option = task_cfg.option
    abspath_data_cfg(data_cfg)
    
    check_cfg(cfg)
    model_cfg.load_from = args.load_from
    abspath_model_cfg(cfg, model_cfg, data_cfg)
    set_output_dir(cfg)
    return cfg
    # common config

def set_output_dir(cfg: Kg2textConfig):
    """
    checkpoint: checkpoint_{task}_{data}/{time_stamp}/.log, .txt, .pt, .cfg
    """
    task_cfg = cfg.task
    encoder_type, decoder_type = cfg.model.encoder_type, cfg.model.decoder_type
    folder = "{}_{}_{}_{}".format(encoder_type, decoder_type, task_cfg.option, task_cfg.dataset)
    cfg.common.output_dir = get_kg2text_abs_path("checkpoint", cfg.common, folder)

    # now, data_cfg.encoder, data_cfg.dataset, model_cfg.n_head, model_cfg.n_layers, tokenization,
    # data_cfg.max_fact, task_cfg.additional, task_cfg.percent)
    # common.output_dir, common.loss_log_file, common.res_file
    return cfg.common.output_dir

def removeprefix(self: str, prefix: str) -> str:
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]

def get_father_path(path, label):

    father_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    grandfather_path = os.path.abspath(os.path.join(father_path, ".."))

    return father_path if label == "father" else grandfather_path


def paraFromKgpt3(model_new,model_trained):
    if not isinstance(model_new, OrderedDict):
        dict_new = model_new.state_dict()
    else:
        dict_new = model_new
    if not isinstance(model_trained, OrderedDict):
        dict_trained = model_trained.state_dict()
    else:
        dict_trained = model_trained
    list_new = list(dict_new.keys())
    list_trained = list(dict_trained.keys())
    for key_new in list_new:
        # suffix = ".".join(key_new.split(".")[1:])
        suffix = key_new
        if "output_projection" not in suffix:
            continue
        elif "output_projection" in suffix:
            suffix = removeprefix(suffix, "output_projection.")
        matching = [s for s in list_trained if suffix in s]

        def get_key_trained(matching, suffix):
            if len(matching) > 1:
                equal = [s for s in list_trained if suffix == s]
                if len(equal) != 1:
                    raise Exception
                else:
                    key = suffix
            elif len(matching) == 1:
                key = matching[0]
            return key

        key_trained = get_key_trained(matching, suffix)
        para_new, para_trained = dict_new[key_new], dict_trained[key_trained]

        assert para_new.shape == para_trained.shape
        if isinstance(para_trained, Parameter):
            para_trained = para_trained.data
        dict_new[key_new].copy_(para_trained)

    return dict_new

def para2kgpt_mbart(model_new, model_trained):
    if not isinstance(model_new, OrderedDict):
        dict_new = model_new.state_dict()
    else:
        dict_new = model_new
    if not isinstance(model_trained, OrderedDict):
        dict_trained = model_trained.state_dict()
    else:
        dict_trained = model_trained
    list_new = list(dict_new.keys())
    list_trained = list(dict_trained.keys())

    count = 0
    for key_new in list_new:
        #suffix = ".".join(key_new.split(".")[1:])
        suffix = key_new
        if "proj_to" in suffix:
            continue
        matching = [s for s in list_trained if suffix in s]
        if len(matching) > 1:
            equal = [s for s in list_trained if suffix == s]
            if len(equal) != 1:
                raise Exception
            else:
                key_trained = key_new
        elif len(matching) == 1:
            key_trained = matching[0]
        para_new, para_trained = dict_new[key_new], dict_trained[key_trained]
        assert para_new.shape == para_trained.shape
        if isinstance(para_trained, Parameter):
            para_trained = para_trained.data
        dict_new[key_new].copy_(para_trained)
        count += 1
    print(count)
    #model_new.load_state_dict(dict_new)
    c = 0
    if c == 1:
        L = len(list_new)-2
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
            dict_new[name].copy_(param)

def copy_para(label, reloaded, model, cfg: Kg2textModelConfig):
    """
    1. exactly the same: just model.load_state_dict
    2. missing part: strict = False
    3. alias: kgpt_kgpt_encoder -> kgpt_kgpt_fairseq_encoder (word_embedding layer)
    name: 
    """
    if (cfg.encoder_type, cfg.decoder_type) == ("kgpt", "mbart50"):
        pass


def saveMyModel(model, label, cfg):
    if label == "encoder":
        save_file = get_kg2text_abs_path("encoder_save", cfg, cfg.save_encoder_file)
        cfg.save_encoder_file = save_file
        torch.save(model.encoder.state_dict(), save_file)
    elif label == "decoder":
        save_file = get_kg2text_abs_path("decoder_save", cfg, cfg.save_decoder_file)
        cfg.save_decoder_file = save_file
        torch.save(model.decoder.state_dict(), save_file)

def printMyModel(mymodel):
    if not isinstance(mymodel, OrderedDict):
        dict = mymodel.state_dict()
    else:
        dict = mymodel
    for k, v in dict.items():
        print("k=", k, ";", "v.size = ", v.size())
    print("Running Transformer with copy gate")
# TransformerDecoder from Wav2Vec

def save_my_cfg(config, file_name):
    with open(file_name, "wb") as fp:
        OmegaConf.save(config=config, f=fp.name)
    print("save config file as {} file".format(file_name))

def load_my_cfg(file_name):
    cfg = OmegaConf.load(file_name)
    print("load yaml file from {}".format(file_name))
    return cfg

def myConfigUpdate(cfg_in, cfg_new):
    dir_in = cfg_in.__dir__()
    for key in dir_in:
        val = cfg_in[key]
        if isinstance(val, DictConfig):
            myConfigUpdate(val, getattr(cfg_new, key))
        else:
            setattr(cfg_new, key, val)
        # TODO: integrate path conversion
    return cfg_new

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", type=bool, default=False, help="Ture for using CPU else False")
    parser.add_argument('--option', type=str, default="train",
                        help="whether to train or test the model", choices=['train', 'test', 'challenge', 'visualize', 'LM', 'compute_bleu', 'few_shot'])
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--train_path', type=str, default="/dataset", help="absolute parent path of dataset")
    parser.add_argument('--test_path', type=str, default="/efs-storage/Kg2text/KGPT/dataset", help="absolute parent path of dataset")
    parser.add_argument('--challenge_path', type=str, default="/efs-storage/Kg2text/KGPT/dataset", help="the embedding dimension")
    parser.add_argument('--knowledge_path', type=str, default="/efs-storage/KGPT/preprocess/knowledge-full.json", help="specify when train on wikidata, knowledge of the entities")
    parser.add_argument('--epochs', type=int, default=10, help="epoches")
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help="the embedding dimension") # TODO: experiment set: 1
    parser.add_argument('--n_head', type=int, default=8, help="the embedding dimension")
    parser.add_argument('--tokenizer_dir', type=str, default='/efs-storage/KGPT/GPT2_tokenizer', help="tokenizer loaded from")
    parser.add_argument('--n_layers', type=int, default=6, help="the embedding dimension")
    parser.add_argument('--max_len', type=int, default=8, help="the embedding dimension")
    parser.add_argument('--dataset', type=str, default='', help="the embedding dimension")
    parser.add_argument('--config', type=str, default='', help="config file for the embedding layer, knowledge_config.json file")
    parser.add_argument('--embedding_path', type=str, default='', help="pretrained BERT embedding weight")
    parser.add_argument('--max_enc_len', type=int, default=640, help="maximum length of the encoding part")
    parser.add_argument('--max_dec_len', type=int, default=72, help="maximum length of the decoding part")
    parser.add_argument('--output_dir', type=str, default='checkpoint', help="maximum length of the decoding part")
    parser.add_argument('--logging_steps', type=int, default=20, help="maximum length of the decoding part")
    parser.add_argument('--printing_steps', type=int, default=10, help="steps per printing") # TODO: experiment set: 1
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="maximum length of the decoding part")
    parser.add_argument('--max_fact', type=int, default=12, help="maximum length of the decoding part")
    parser.add_argument('--max_entity', type=int, default=12, help="maximum length of the decoding part")
    parser.add_argument('--starting_epoch', type=int, default=0, help="maximum length of the decoding part")
    parser.add_argument('--load_from', type=str, default="/Users/xianjiay/workplace/Kg2text/model_ep14.pt", help="maximum length of the decoding part")
    parser.add_argument('--num_workers', type=int, default=8, help="maximum length of the decoding part")
    parser.add_argument('--beam_size', type=int, default=2, help="the embedding dimension")
    parser.add_argument('--bleu', type=int, default=4, help="the embedding dimension")
    parser.add_argument('--hidden_size', type=int, default=None, help="the embedding dimension")
    parser.add_argument('--finetune', action='store_true', help='Which experiment you are doing')
    parser.add_argument('--additional', type=str, default="", help='Which experiment you are doing')
    parser.add_argument('--unforbid_duplicate', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--encoder', type=str, required=True, choices=['sequence', 'graph', 'graph_finegrained'], help='Which experiment you are doing')
    parser.add_argument('--lower_case', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--copy_loss', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--percent', default=1.0, type=float, help='Which experiment you are doing')
    args = parser.parse_args()

    return args

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if 'post_word_emb' in name:
            print("Sin/Cos embedding does not need to be reloaded")
            continue
        if 'entity_embeddings' in name or 'triple_embeddings' in name:
            if param.shape != own_state[name].shape:
                print("Reinitializing the weight for {}".format(name))
                continue
            else:
                pass
        if isinstance(param, Parameter): # TODO CLASS torch.nn.parameter.Parameter
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def get_pretrained_mbart50(cfg, component):
    # decoder = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from).model.decoder
    model_file = "model.pt"
    dictionary_file = "dict.mbart50.txt"
    data_path = "mbart50"
    model_file = get_kg2text_abs_path("model_mbart", cfg, model_file)
    dictionary_file = get_kg2text_abs_path("dict", cfg, dictionary_file)
    data_path = get_kg2text_abs_path("model_mbart", cfg, data_path)

    cp = model_file
    mbart, cfg, task = checkpoint_utils.load_model_ensemble_and_task([cp], arg_overrides={
        'lang_dict': dictionary_file,  # ML50_langs.txt/ dict.mbart50.txt
        'data': data_path})
    """
    cp = '/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50/model.pt'
    mbart, cfg_mbart, task = checkpoint_utils.load_model_ensemble_and_task([cp], arg_overrides={
        'lang_dict': '/media/MyDataStor1/jxian/efs-storage/Kg2text/tokenizer/mbart50/dict/dict.txt', # ML50_langs.txt
        'data': '/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50'})
    """
    if component == "encoder":
        return mbart[0].encoder
    elif component == "decoder":
        return mbart[0].decoder
    elif component == "model":
        return mbart[0]
    elif component == "cfg":
        return cfg
    elif component == "task":
        return task


def load_mbart50_whole(cfg):
    cfg.checkpoint_file = get_kg2text_abs_path("model_mbart50", cfg, cfg.checkpoint_file)
    cp = cfg.checkpoint_file
    data_path = get_kg2text_abs_path("model_mbart50", cfg, "")
    lang_dict_path = os.path.join(data_path, "ML50_langs.txt")
    # cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)
    # "lang_dict": "/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50/ML50_langs.txt",
    # "data": "/media/MyDataStor1/jxian/efs-storage/Kg2text/model/mbart50"
    mbart50, cfg, task = checkpoint_utils.load_model_ensemble_and_task([cp], arg_overrides={
        "lang_dict": lang_dict_path,
        "data": data_path
    })
    return mbart50[0], cfg, task


def lang_tag(prepend_lang: bool, lang, text):
    lang_tag_template = "[{}]"
    if lang is not None and prepend_lang:
        # assume the tgt_dict already have language tags in the dict, e.g. mBART50 task loaded dictionary
        # so prepend the tag direct and have encode_line convert it to ID.
        text_tagged = "{lang_tag} {x}".format(
            lang_tag=lang_tag_template.format(lang),
            x=text)
    else:
        return text

    return text_tagged

def load_bpe_tokenizer(label, cfg: Kg2textDataSetConfig):
    if label == "fairseq_gpt2":
        cfg.fairseq_gpt2.gpt2_encoder_json = get_kg2text_abs_path("bpe", cfg.fairseq_gpt2, cfg.fairseq_gpt2.gpt2_encoder_json)
        cfg.fairseq_gpt2.gpt2_vocab_bpe = get_kg2text_abs_path("bpe", cfg.fairseq_gpt2, cfg.fairseq_gpt2.gpt2_vocab_bpe)
        tokenizer = GPT2BPE(cfg.fairseq_gpt2)

    elif label in ["sentencepiece", "mbart50"]:
        cfg.sentenpiece.sentencepiece_model = get_kg2text_abs_path("bpe", cfg.sentenpiece,
                                                                   cfg.sentenpiece.sentencepiece_model)
        tokenizer = SentencepieceBPE(cfg.sentenpiece)
    elif label == "kgpt_gpt2":
        cfg.kgpt_gpt2.tokenizer_dir = get_kg2text_abs_path("bpe", cfg.kgpt_gpt2, cfg.kgpt_gpt2.tokenizer_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.kgpt_gpt2.tokenizer_dir)
    else:
        raise NotImplementedError

    return tokenizer


def tokenize_text_test(cfg, text, source: str = "tgt", lang=None):
    lang_tag_template = "[{}]"
    src_bpe = load_bpe_tokenizer("fairseq_gpt2", cfg)
    # TODO path collection
    cfg.vocab_file_src = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_src)
    cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)
    src_dict = Dictionary.load(cfg.vocab_file_src)
    tgt_bpe = load_bpe_tokenizer("sentencepiece", cfg)
    tgt_dict = Dictionary.load(cfg.vocab_file_tgt)
    src_lang_tag_template = lang_tag_template
    tgt_lang_tag_template = lang_tag_template
    prepend_lang_tag = True

    # {"[ENT]": 50257, "[PRED]": 50258, "[SUB]": 50259, "[TRIPLE]": 50260, "[EOS]": 50261, "[SOS]": 50262, "[PAD]": 50263}
    special_symbols = ["[ENT]", "[PRED]", "[SUB]", "[TRIPLE]"]
    for tag in special_symbols:
        tgt_dict.add_symbol(tag, 1, False)

    # 'main ingredients', 'Binignit'

    # [TRIPLE] [PRED] description [SUB] {} [TRIPLE]
    #text1 = "[ENT] Sweet potato [TRIPLE] [PRED] description [SUB] Sweet potato [TRIPLE]  [PRED] main ingredients [SUB] Binignit [TRIPLE]"
    text1 = "[ENT] Sweet potato"
    text2 = "Sweet potato"
    tokenized_text1 = tgt_bpe.encode(text1)
    tokenized_text2 = tgt_bpe.encode(text2)
    tokenized_text3 = "[en_XX] " + tgt_bpe.encode(text2)
    text_done1 = tgt_dict.encode_line(tokenized_text1, add_if_not_exist=False, append_eos=True)
    text_done2 = tgt_dict.encode_line(tokenized_text2, add_if_not_exist=False, append_eos=True)
    text_done3 = tgt_dict.encode_line(tokenized_text3, add_if_not_exist=False, append_eos=True)
    with open("decoded_test.txt", "a") as f:
        f.write(tokenized_text3 + "\n")
        f.close()
    token = []
    with open("decoded_test.txt") as f:
        for line in f:
            # For Python3, use print(line)
            token.append(tgt_dict.encode_line(line))
            if 'str' in line:
                break

    tokenized_tgt = tgt_bpe.encode(text)
    text_done_tgt = tgt_dict.encode_line(
        tokenized_tgt, add_if_not_exist=False, append_eos=True
    ).long()
    tokenizer = GPT2Tokenizer.from_pretrained("/media/MyDataStor1/jxian/efs-storage/Kg2text/GPT2_tokenizer")
    banwords = ['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they']
    banwords1 = tokenizer.convert_tokens_to_ids(['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they'])
    text_done = tokenizer.convert_tokens_to_ids(text2)
    #banwords_src = tokenized_src.encode(" ".join(banwords))

    return text_done1


def load_dictionary(cfg: Kg2textDataSetConfig, label=None, symbols_added=None):
    if label == "src":
        cfg.vocab_file_src = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_src)
        src_dict = Dictionary.load(cfg.vocab_file_src)
        if symbols_added:
            for sym in symbols_added:
                src_dict.add_symbol(sym, 1)
        return src_dict
    elif label == "tgt":
        cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)
        tgt_dict = Dictionary.load(cfg.vocab_file_tgt)
        if symbols_added:
            for sym in symbols_added:
                tgt_dict.add_symbol(sym, 1)
        return tgt_dict
    else:
        cfg.vocab_file_src = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_src)
        src_dict = Dictionary.load(cfg.vocab_file_src)
        if symbols_added:
            for sym in symbols_added:
                src_dict.add_symbol(sym, 1)
        cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)
        tgt_dict = Dictionary.load(cfg.vocab_file_tgt)
        if symbols_added:
            for sym in symbols_added:
                tgt_dict.add_symbol(sym, 1)
        return {"src_dict": src_dict, "tgt_dict": tgt_dict}

class mytokenizer:

    def __init__(self, bpe, dict):
        self.bpe, self.dict = bpe, dict
        self.pad_token_id = self.dict.pad_index
        self.bos_token_id = self.dict.bos_index
        self.eos_token_id = self.dict.eos_index
        self.unk_token_id = self.dict.unk_index
    @classmethod
    def build_tokenizer(cls, cfg, label):
        if label == "src":
            bpe = load_bpe_tokenizer(cfg.tokenizer_src, cfg)
            dict = load_dictionary(cfg, "src")
        elif label == "tgt":
            bpe = load_bpe_tokenizer(cfg.tokenizer_tgt, cfg)
            dict = load_dictionary(cfg, "tgt")
        else:
            raise NotImplementedError
        return cls(bpe, dict)

    def encode(self, text: str):
        text_tokenized = self.bpe.encode(text)
        text_done = self.dict.encode_line(text_tokenized, add_if_not_exist=False, append_eos=False)
        #l = self.bpe.bpe("greenhand")
        return text_done

    def encode_lang_tag(self, text, prepend_lang_tag=False, lang=None, append_eos=False):
        text_tokenized = self.bpe.encode(text)
        if lang is not None and prepend_lang_tag:
            # assume the tgt_dict already have language tags in the dict, e.g. mBART50 task loaded dictionary
            # so prepend the tag direct and have encode_line convert it to ID.
            text_tokenized = self.lang_tagging(text_tokenized, lang)
        text_done = self.dict.encode_line(text_tokenized, add_if_not_exist=False, append_eos=append_eos)
        return text_done

    def decode(self, tensor, prepend_lang_tag=False, lang=None, **kwargs):
        text_tokenized = self.dict.string(tensor)
        text = self.bpe.decode(text_tokenized)
        return text

    def lang_tagging(self, text, lang):
        lang_tag_template = "[{}]"
        # prepend lang_tag after bpe encoding
        text_tagged = "{lang_tag} {tokenized}".format(
            lang_tag=lang_tag_template.format(lang),
            tokenized=text)
        return text_tagged


def get_tokenizer(cfg: Kg2textDataSetConfig, label):
    # TODO add Kg2textTokenConfig
    tokenizer_type = getattr(cfg, "tokenizer_" + label)
    if tokenizer_type in ["sentencepiece", "mbart", "mbart50"]:
        tokenizer = mytokenizer.build_tokenizer(cfg, label)
    elif tokenizer_type == "kgpt_gpt2":
        token_cfg = getattr(cfg, tokenizer_type)
        token_cfg.tokenizer_dir = get_kg2text_abs_path("bpe", token_cfg, token_cfg.tokenizer_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(token_cfg.tokenizer_dir)
    elif tokenizer_type == "fairseq_gpt2":
        tokenizer = GPT2BPE(getattr(cfg, tokenizer_type))
    else:
        raise NotImplementedError

    return tokenizer



def experiment(task_cfg, data_cfg, model_cfg, common, token_cfg, src_dict, tgt_dict):
    #loaded = load_kgpt_model(None, model_cfg)
    #model, _, _ =load_mbart50_whole(model_cfg)
    #save_component("encoder", model.encoder, model_cfg)
    build_fairseq_dataset(data_cfg, "test", src_dict, tgt_dict)
    #torch.save(model.decoder.state_dict(), model_cfg.save_decoder_file)

def save_component(component, model, cfg: Kg2textModelConfig):
    save_file = "{encoder_type}_{decoder_type}_{component}.pt".format(
        encoder_type = cfg.encoder_type, decoder_type =cfg.decoder_type, component=component)
    
    abs_save_file = get_kg2text_abs_path(component+"_save", cfg, save_file)
    torch.save(model.state_dict(), abs_save_file)


def build_fairseq_dataset(cfg, split, src_dict, tgt_dict):

    cfg.option = split
    from DataLoader import FairseqTranferDataset
    data = FairseqTranferDataset(cfg, src_dict, tgt_dict)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(data)
    L = len(data)
    input_file = split + "_input.txt"
    label_file = split + "_label.txt"
    input_file = get_kg2text_abs_path("data", cfg, input_file)
    label_file = get_kg2text_abs_path("data", cfg, label_file)
    for i in range(L):
        x = data[i]

        with open(input_file, "a") as f1:
            f1.write(x["text_bped"]+"\n")

        with open(label_file, "a") as f2:
            f2.write(x["sent_bped"]+"\n")

    f1.close()
    f2.close()

def load_kgpt_state_dict(model_cfg):
    model_cfg.load_from = "model_seq.pt"
    model_cfg.load_from = get_kg2text_abs_path("model_kgpt", model_cfg, model_cfg.load_from)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        reloaded = torch.load(model_cfg.load_from, map_location=torch.device('cpu'))
    elif device.type == "cuda":
        reloaded = torch.load(model_cfg.load_from)
        print("reloaded:", "reloaded.type()" )
    #load_my_state_dict(model, reloaded)
    return reloaded
        #model.load_state_dict(reloaded)
        #print(model.parameters())
    logger.info("Loading model from {}".format(model_cfg.load_from))

def para_copy_kgpt_kgpt_encoder(model_new, model_trained):
    if not isinstance(model_new, OrderedDict):
        dict_new = model_new.state_dict()
    else:
        dict_new = model_new
    if not isinstance(model_trained, OrderedDict):
        dict_trained = model_trained.state_dict()
    else:
        dict_trained = model_trained
    list_new = list(dict_new.keys())
    list_trained = list(dict_trained.keys())

    count = 0
    for key_new in list_new:
        
        #suffix = ".".join(key_new.split(".")[1:])
        if "proj_to" in key_new:
            continue
        suffix = key_new
        matching = [s for s in list_trained if suffix in s]
        assert len(matching) == 1
        
        key_trained = matching[0]
        para_new, para_trained = dict_new[key_new], dict_trained[key_trained]
        assert para_new.shape == para_trained.shape
        if isinstance(para_trained, Parameter):
            para_trained = para_trained.data
        dict_new[key_new].copy_(para_trained)
        count += 1
    print(count)

def para_copy_mbart50t_mbart50t(model_new, model_trained):
    # for the keys with different shapes of paras
    # only loaded for same keys with same shapes of paras
    if not isinstance(model_new, OrderedDict):
        dict_new = model_new.state_dict()
    else:
        dict_new = model_new
    if not isinstance(model_trained, OrderedDict):
        dict_trained = model_trained.state_dict()
    else:
        dict_trained = model_trained
    list_new = list(dict_new.keys())
    list_trained = list(dict_trained.keys())

    count = 0
    for key_new in list_new:
        
        #suffix = ".".join(key_new.split(".")[1:])
        if dict_trained.get(key_new) is None:
            continue
        key_trained = key_new
        
        para_new, para_trained = dict_new[key_new], dict_trained[key_trained]
        if para_new.shape != para_trained.shape:
            print(key_new)
            continue
        if isinstance(para_trained, Parameter):
            para_trained = para_trained.data
        dict_new[key_new].copy_(para_trained)
        count += 1
    print(count)


def paraFromKgpt3(model_new,model_trained):
    if not isinstance(model_new, OrderedDict):
        dict_new = model_new.state_dict()
    else:
        dict_new = model_new
    if not isinstance(model_trained, OrderedDict):
        dict_trained = model_trained.state_dict()
    else:
        dict_trained = model_trained
    list_new = list(dict_new.keys())
    list_trained = list(dict_trained.keys())
    for key_new in list_new:
        # suffix = ".".join(key_new.split(".")[1:])
        suffix = key_new
        if "output_projection" not in suffix:
            continue
        elif "output_projection" in suffix:
            suffix = removeprefix(suffix, "output_projection.")
        matching = [s for s in list_trained if suffix in s]

        def get_key_trained(matching, suffix):
            if len(matching) > 1:
                equal = [s for s in list_trained if suffix == s]
                if len(equal) != 1:
                    raise Exception
                else:
                    key = suffix
            elif len(matching) == 1:
                key = matching[0]
            return key

        key_trained = get_key_trained(matching, suffix)
        para_new, para_trained = dict_new[key_new], dict_trained[key_trained]

        assert para_new.shape == para_trained.shape
        if isinstance(para_trained, Parameter):
            para_trained = para_trained.data
        dict_new[key_new].copy_(para_trained)

    return dict_new