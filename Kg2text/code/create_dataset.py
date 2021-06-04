import omegaconf
from sentencepiece import _add_snake_case
import torch
import json
import random
from torch.utils.data import Dataset
from fairseq.data import Dictionary
import logging
import os
from myutils import load_bpe_tokenizer
import argparse


import logging
from typing import Dict, List, Optional, OrderedDict, Tuple, Any
from fairseq import utils, dataclass
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig
from omegaconf import MISSING, OmegaConf, II
from dataclasses import dataclass, field
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

random.seed(0)
logger = logging.getLogger(__name__)
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

# cfg, tasl_cfg, model_cfg, token_cfg

def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return


@dataclass
class Kg2KgDatasetConfig(FairseqDataclass):
    # TODO: maybe just fix the directory structure to force it to be relative to II("task.data")
    # II(): dataset task.dataset
    fairseq_gpt2: GPT2BPEConfig = GPT2BPEConfig()
    sentenpiece: SentencepieceConfig = SentencepieceConfig()

    option: str = field(default="no_tag", metadata={"help": "."}) # == task.option
    split: str = field(default="train", metadata={"help": "."})
    dataset: str = field(default="webnlg", metadata={"help": "."})
    tokenizer_src: str = field(default="kgpt_gpt2", metadata={"help": "source tokenizer type"})
    vocab_file_src: str = field(default="dict.txt", metadata={"help": "."})
    #bpe_tokenizer_src: GPT2BPEConfig = GPT2BPEConfig()
    #vocab_file_src: str = field(default= "dict.gpt2.json", metadata={"help": "source dictionary file"})

    tokenizer_tgt: str = field(default="mbart50", metadata={"help": "target tokenizer type"})
    vocab_file_tgt: str = field(default="dict.mbart50.txt", metadata={"help": "."})
    #bpe_tokenizer_tgt: SentencepieceConfig = SentencepieceConfig()
    #vocab_file_tgt: str = field(default="dict.mbart50.json", metadata={"help": "target dictionary file"})

    # dataset: str = field(default="wikidata", metadata={"help": "choose dataset"})
    # dataset: Optional[str] = II("task.dataset")
    
    #train_file_path: str = field(default="", metadata={"help": "kg triples training dataset path relative to which to the project"})
    train_file: str = field(default="", metadata={"help": "train subset file name"})
    test_file: str = field(default="", metadata={"help": "val subset file name"})
    eval_file: str = field(default="", metadata={"help": "test subset file name"} )

    fixed_src_length: bool = field(default=False, metadata={"help": "."})
    fixed_tgt_length: bool = field(default=False, metadata={"help": "."})
    src_length: int = field(default=760, metadata={"help": "."})
    tgt_length: int = field(default=50, metadata={"help": "."})

    # multilingual related
    src_lang_tag_template: str = field(default="[{}]", metadata={"help": "."})
    tgt_lang_tag_template: str = field(default="[{}]", metadata={"help": "."})
    src_lang: str = field(default="", metadata={"help": "."})
    tgt_lang: str = field(default="", metadata={"help": "."})
    prepend_src_lang_tag: bool = field(default=False, metadata={"help": "."})
    prepend_tgt_lang_tag: bool = field(default=False, metadata={"help": "."})

    # with added tags
    # kgpt: enc/dec + lang_tags / mbart50: dec + special tags
    src_wtags: bool = field(default=True, metadata={"help": "."})
    tgt_wtags: bool = field(default=True, metadata={"help": "."})
    # Kgpt seq related
    max_entity: int = field(default=50, metadata={"help": "."})
    max_fact: int = field(default=50, metadata={"help": "."})
    percent: float = field(default=1.0, metadata={"help": ""})

    forbid_duplicate_relation: bool = field(default=True, metadata={"help": "."})
    lower_case: bool = field(default=False, metadata={"help": "."})
    encoder_arch: str = field(default="sequence", metadata={"help": "."})
    knowledge_file: str = field(default="", metadata={"help": "."})

    sep_symbol: str = field(default="|", metadata={"help": "specify a seperate symbol to seperate ent|pred|sub"})

    

    #max_entity: int = field(default=12, metadata={"help": "number of workers"})
    #max_fact: int = field(default=8, metadata={"help": "number of workers"})

    #forbid_duplicate_relation: bool = field(default=True, metadata={"help": "forbid_duplicate_relation..??"})
    #knowledge_path: str = field(default="/knowledge-full.json",
    #                           metadata={"help": "for wikidata training only, augmented entities knowledge"})
    #shuffle: bool = field(
    #    default=True,
    #    metadata={"help": "if set, shuffle dataset samples before batching"},
    #)

class Kg2TextDataset(Dataset):
    def __init__(self, ):
        pass

class Text2TextDataset(Dataset):
    def __init__(self, ):
        pass


class Kg2KgDataset(Dataset):
    def __init__(self,cfg, split, src_dict, tgt_dict, args):
        super(Kg2KgDataset, self).__init__()

        #cfg = cfg.cfg
        #assert isinstance(cfg, Kg2KgDatasetConfig)
        ent = "[ENT]"
        pred = "[PRED]"
        sub = "[SUB]"
        triple = "[TRIPLE]"
        self.tags = [ent, pred, sub, triple]
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_bpe = SentencepieceBPE(cfg.sentencepiece)
        self.tgt_bpe = self.src_bpe
        self.option = cfg.option

        self.src_lang_tag = cfg.src_lang_tag_template.format(cfg.src_lang)
        self.tgt_lang_tag = cfg.tgt_lang_tag_template.format(cfg.tgt_lang)
        self.lang_entity_format = cfg.src_lang_tag_template.format(cfg.src_lang) + " {}"
        self.entity_format = ent + " {}"
        # words_tagged = '[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower())
        self.des_format = triple + " " + pred + " description " + sub + " {} " + triple
        self.rel_format = pred + " {} " + sub + " {} " + triple
       
        self.ent_tagged_format = ent + " {} "
        self.ent_text_only_format = "{} "
        self.ent_simple_format = "{} [SEP] "

        self.triple_unseperate_tagged_format = pred + " {} " + sub + " {} " + triple
        self.triple_seperate_tagged_format = pred + " {} " + sub + " {} " + triple
        self.triple_text_only_format = "{} {} "
        self.triple_simple_format = "{} [SEP] {} [SEP]"
        
        self.triple_simple_sep = "[SEP]"

        if cfg.tokenizer_src in ["sentencepiece", "mbart", "mbart50", "mbart50t"]:
            self.des_format = triple + " " + pred + " " + self.src_bpe.encode(
                "description") + " " + sub + " {} " + triple
        elif "gpt" in cfg.tokenizer_src:
            self.des_format = triple + " " + pred + " description " + sub + " {} " + triple

        if cfg.tokenizer_tgt in ["sentencepiece", "mbart", "mbart50", "mbart50t"]:
            self.lang_text_format = cfg.tgt_lang_tag_template.format(cfg.tgt_lang) + " {} " + self.tgt_dict.eos_word
            self.text_format = self.tgt_dict.bos_word + " {} " + self.tgt_dict.eos_word
        elif "gpt" in cfg.tokenizer_tgt:
            self.lang_text_format = cfg.tgt_lang_tag_template.format(cfg.tgt_lang) + " {} " + self.tgt_bpe.eos_token
            self.text_format = self.tgt_bpe.bos_token + " {} " + self.tgt_bpe.eos_token

        self.max_entity = cfg.max_entity
        self.max_fact = cfg.max_fact
        self.forbid_duplicate_relation = cfg.forbid_duplicate_relation
        self.percent = cfg.percent
        self.lower_case = cfg.lower_case

        self.fixed_src_length = cfg.fixed_src_length
        self.fixed_tgt_length = cfg.fixed_tgt_length

        self.src_tokenizer = cfg.tokenizer_src #
        self.src_max = cfg.src_length
        self.src_lang_tag_template = cfg.src_lang_tag_template
        self.src_enc_type = cfg.encoder_arch
        self.src_lang = cfg.src_lang
        self.prepend_src_lang_tag = cfg.prepend_src_lang_tag
        self.src_tagged = cfg.src_tagged
        self.kg_tagged = cfg.kg_tagged
        self.text_tagged = cfg.text_tagged

        self.tgt_tokenizer = cfg.tokenizer_tgt
        self.tgt_max = cfg.tgt_length
        self.tgt_lang_tag_template = cfg.tgt_lang_tag_template
        self.tgt_lang = cfg.tgt_lang
        self.prepend_tgt_lang_tag = cfg.prepend_tgt_lang_tag

        self.sep= cfg.sep_symbol

        self.seperate=args.seperate
        self.text_only=args.text_only
        self.simple=args.simple
        self.tagged=args.tagged
        self.add_eos=args.add_eos
        self.add_bos=args.add_bos

        file_path = getattr(cfg, split+"_file", "")
        #knowledge_path = getattr(cfg, knowledge_file, "")
        if file_path == "":
            logger.warning("Dataset for task: ", cfg.option, " is not specified, using eval dataset.")
            file_path = cfg.valid_file

        with open(file_path, 'r') as f:
            self.data = json.load(f)
            print("Loaded data from ", file_path, " for task ", cfg.split)
        f.close()
        if cfg.percent > 1:
            self.data = self.data[:int(self.percent)]
        else:
            selected_size = int(len(self.data) * self.percent)
            self.data = self.data[:selected_size]
        
        if cfg.dataset == "kgtext_wikidata":
            with open(cfg.knowledge_file, 'r') as f:
                self.knowledge = json.load(f)
                print("Loaded data from ", cfg.knowledge_file, " for task ")

    def tokenize_text(self, text, tokenizer, type):
        if type in ["sentencepiece", "mbart", "mbart50"]:
            tokenized_text = tokenizer.encode_line(text)
        elif type in ["kgpt", "gpt2"]:
            tokenized_text = tokenizer.encode(text)
        else:
            raise NotImplementedError
        return tokenized_text

    def lang_tagging(self, text):
        # prepend lang_tag after bpe encoding
        text_tagged = "{lang_tag} {tokenized}".format(
            lang_tag=self.src_lang_tag_template.format(self.src_lang),
            tokenized=text)

        return text_tagged


    def format_triples(self, entity, triples_setting):
        # seperate/tagged/tokenized
        # seperate: break each sample into a set of triples
        # tagged: tiples will be tagged [KG] [ENT] Sweet potato [PRED] main ingrredients [SUB] Binignit [TRIPLE]
        # tokenized: words will be tokenized by bpe tokenizer
        # entity: ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
        
        seperate, tokenized, text_only, simple, tagged, lower_case, skip_des, lang = triples_setting.seperate, \
            triples_setting.tokenized, triples_setting.text_only, triples_setting.simple, triples_setting.tagged, triples_setting.lower_case, triples_setting.skip_des, triples_setting.lang

        if lower_case:
            entity0, entity1 = entity[0].lower(), entity[1].lower()
        else:
            entity0, entity1 = entity[0], entity[1]
        
        if tokenized:
            entity0, entity1 = self.src_bpe.encode(entity0),\
                self.src_bpe.encode(entity1)

        
        if text_only:
            ent_part = self.ent_text_only_format.format(entity0)
        elif simple:
            ent_part = self.ent_simple_format.format(entity0)
        elif tagged:
            ent_part = self.ent_tagged_format.format(entity0)


        triple_part = ent_part
        added = set()
        triples = []
        if seperate == False:
            for fact in entity[2]:
                if self.forbid_duplicate_relation and fact[0] in added:
                    continue
                else:
                    if lower_case:
                        pred, sub = fact[0].lower(), fact[1].lower()
                    else:
                        pred, sub = fact[0], fact[1]

                    if tokenized:
                         pred, sub = self.src_bpe.encode(pred),\
                self.src_bpe.encode(sub)
                    # ["Sweet potato", "main  ingredients", "Binignit"]
                    
                    if text_only:
                        triple_part += self.triple_text_only_format.format(pred, sub)
                    elif simple:
                        triple_part += self.triple_simple_format.format(pred, sub)
                    elif tagged:
                        triple_part += self.triple_unseperate_tagged_format.format(pred, sub)
                    else:
                        raise NotImplementedError

                if len(added) >= self.max_fact:
                    break
            
            # triples: ["Sweet potato | main ingredients | Binignit", "Sweet potato | yyy | zzz" ]
            return triple_part
            
        elif seperate == True:
            for fact in entity[2]:
                triple_part = ""
                if self.forbid_duplicate_relation and fact[0] in added:
                    continue
                else:
                    if lower_case:
                        pred, sub = fact[0].lower(), fact[1].lower()
                    else:
                        pred, sub = fact[0], fact[1]

                    if tokenized:
                         pred, sub = self.src_bpe.encode(pred),\
                self.src_bpe.encode(sub)
                    # ["Sweet potato", "main  ingredients", "Binignit"]
                    
                    if text_only:
                        triple_part = ent_part + self.triple_text_only_format.format(pred, sub)
                    elif simple:
                        triple_part = ent_part + self.triple_simple_format.format(pred, sub)
                    elif tagged:
                        triple_part = ent_part + self.triple_seperate_tagged_format.format(pred, sub)
                    else:
                        raise NotImplementedError
                    
                    triples.append(triple_part)

                if len(added) >= self.max_fact:
                    break
            
            # triples: ["Sweet potato | main ingredients | Binignit", "Sweet potato | yyy | zzz" ]
            return triple_part


    def linearize_bpe(self, entity, lang=None, lower_case=False ):
        if lower_case:
            entity0, entity1 = entity[0].lower(), entity[1].lower()
        else:
            entity0, entity1 = entity[0], entity[1]

        entity0, entity1 = self.src_bpe.encode(entity0), self.src_bpe.encode(entity1)
        """
        if self.prepend_src_lang_tag:
            entity_tagged = self.lang_tagging(entity0)
        """

        entity_tagged = self.entity_format.format(entity0)
        # TODO description
        des_tagged = self.des_format.format(entity1)

        #entity_tokenized = self.src_dict.encode_line(entity_tagged, add_if_not_exist=False, append_eos=False)
        #des_tokenized = self.src_dict.encode_line(des_tagged, add_if_not_exist=False, append_eos=False)
        string = entity_tagged + " " + des_tagged

        added = set()
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if lower_case:
                    rel0, rel1 = rel[0].lower(), rel[1].lower()
                else:
                    rel0, rel1 = rel[0], rel[1]
                rel0, rel1 = self.src_bpe.encode(rel0), self.src_bpe.encode(rel1)

                rel_tagged = self.rel_format.format(rel0, rel1)
                #rel_tokenized = self.src_dict.encode_line(rel_tagged, add_if_not_exist=False, append_eos=False)
                string += " " + rel_tagged
                added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        return string


    def sentence_preprocess_bpe(self, sentence, prepend_lang_tag, add_bos=True, add_eos=False, lower_case=False):
        if lower_case:
            sentence =sentence.lower()
        sent_bped = self.tgt_bpe.encode(sentence)
        

        if self.prepend_tgt_lang_tag:
            sent_format = self.lang_text_format
        else:
            sent_format = self.text_format

        if add_bos == False:
            sent_format = sent_format.replace(self.tgt_dict.bos_word + " ", "")
        if add_eos == False:
            sent_format = sent_format.replace(" "+self.tgt_dict.eos_word, "")
        sent_tagged = sent_format.format(sent_bped)
        #sent_tokenized = self.tgt_bpe.encode(sent_tagged)
        return sent_tagged
    
    def get_reference(self, idx, do_lower_case=False):
        if do_lower_case:
            return [_.lower().split(' ') for _ in self.data[idx]['text']]
        else:
            return [_.split(' ') for _ in self.data[idx]['text']]


    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())


    def __len__(self):
        return len(self.data)

    def write2file(self, setting, file_name):
        L = len(self.data)
        
        with open(file_name, "w") as f1:
            #f1.write(x["text_bped"]+"\n"


            for idx in range(L):
                entry = self.data[idx]
                #sentence = random.choice(entry['text']) # TODO
                KBs = entry['kbs'] # set of triples

                # for src_text
                if self.seperate == True:
                    triples = []

                    for i, entity_label in enumerate(KBs): # entity_label: "W1215"
                        if i + 1 >= self.max_entity:
                            break

                        entity = KBs[entity_label] # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                        triple = self.format_triples(entity, setting)
                        triples += triple # triple: list of triples, list of str
                        f1.write(triple + "\n")
                else:
                    triples = ""

                    for i, entity_label in enumerate(KBs): # entity_label: "W1215"
                        if i + 1 >= self.max_entity:
                            break

                        entity = KBs[entity_label] # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                        triple = self.format_triples(entity, setting)
                        triples += triple + " " # triple: list of triples, list of str
                    f1.write(triples + "\n")

            f1.close()

    def write2file2(self, setting, file_name):
        L = len(self.data)
        
        with open(file_name, "w") as f1:
            for i in range(L):
                entry = self.data[i]

                sentence = ' '.join(entry['text'])
                entities = []
                for _ in entry['kblinks']:
                    if _ is not None and _ in self.knowledge and _ not in entities:
                        entities.append(_)

                if self.src_enc_type == 'seq':
                    strings = []
                    entity_ids = []
                    triple_ids = []

                    if 'title' in entry:
                        entity = self.knowledge[entry['title_kb_id']]
                        string, triple_id = self.linearize(entity, self.lower_case)

                        strings += string
                        entity_ids += [0] * len(string)
                        triple_ids += triple_id

                    for i, entity_id in enumerate(entities):
                        if i + 1 >= self.max_entity:
                            break

                        entity = self.knowledge[entity_id]
                        string, triple_id = self.linearize(entity, self.lower_case)
                        # string: all the knowledge(description, rels) tokenized vector
                        # triple_id: indicate different triples in the string

                        strings += string
                        entity_ids += [i + 1] * len(string)
                        triple_ids += triple_id

                    position_ids = list(range(len(strings)))
                    assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

                    input_ids, entity_ids, triple_ids, position_ids = self.entity_truncated(
                        strings, entity_ids, triple_ids, position_ids)

                    sent_tokenized = self.sentence_preprocess(sentence, self.prepend_tgt_lang_tag)
                    output_ids = self.sentence_truncated(sent_tokenized)

                    return input_ids, entity_ids, triple_ids, position_ids, output_ids[:-1], output_ids[1:]
                
                else:
                    raise NotImplementedError

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text']) # TODO
        KBs = entry['kbs'] # set of triples

        # for src_text
        if self.seperate == True:
            triples = []

            for i, entity_label in enumerate(KBs): # entity_label: "W1215"
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label] # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                triple = self.format_triples(entity, seperate=self.seperate, tokenized=self.tokenized, text_only=self.text_only, \
                    simple=self.simple, tagged=self.tagged, lower_case=False,skip_des=True, lang=None)
                triples += triple # triple: list of triples, list of str
            
            return triples


        else:
            pass



def get_dataset_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="kg2kg", help="options: kg2kg, text2text, kg2text")
    parser.add_argument("--seperate", type=bool, default=False, help="True for unsupervised dataset, False for text labelled dataset.")
    parser.add_argument("--text_only", type=bool, default=False, help="Triples and sentences will not be tagged.")
    parser.add_argument("--simple", type=bool, default=False, help="add seperating symbols between parts in a triple and between triples")
    parser.add_argument("--tagged", type=bool, default=True, help="add tags for triples and sentences")
    parser.add_argument("--tokenized", type=bool, default=True, help="apply bpe to words")
    parser.add_argument("--add_eos", type=bool, default="", help="add eos or not")
    parser.add_argument("--add_bos", type=bool, default="", help="add bos or not")
    parser.add_argument("--dataset", type=str, default="webnlg", help="specify dataset")
    parser.add_argument("--config_file", type=str, default="triples_dataset.yaml", help="specify config yaml file")
    parser.add_argument("--setting_file", type=str, default="token_setting.yaml", help="setting to create different types of datasets")
    parser.add_argument("--load_data_dir", type=str, default="", help="specify loading data from data dir")
    parser.add_argument("--save_data_dir", type=str, default="", help="specify saving data dir")
    parser.add_argument("--lang", type=str, default="en_XX", help="lang tag")
    parser.add_argument("--efs", type=str, default="", help="dir of efs")
    
    args = parser.parse_args()

    return args

def get_abs_project_path(project_name="Kg2text", project_path=None):
    # get project_path
    if project_path is None:
        path_cwd = os.path.abspath(os.getcwd())
        lst = path_cwd.split(os.sep)
        if lst[-1] == "fairseq":
            path_cwd += os.sep + project_name
        if project_name not in path_cwd:
            print("specify project dir")
            raise NotImplementedError
        path_cwd_list = path_cwd.split(os.sep)
        idx = path_cwd_list.index(project_name)
        project_path = os.sep.join(path_cwd_list[:idx + 1])
    else:
        lst = project_path.split(os.sep)
        assert project_name == lst[-1]
    
    return project_path


def update_cfg(cfg, args):
    cfg.dataset = args.dataset
    cfg.project_dir = get_abs_project_path()
    cfg.efs = args.efs
    load_data_dir = args.load_data_dir

    for split in ["train", "test", "valid"]:
        #file_abs_path = load_data_dir + os.sep + cfg.dataset + os.sep + "raw_data" + os.sep + cfg.train_file
        file_abs_path = os.path.join(load_data_dir, cfg.dataset, getattr(cfg, split+"_file"))
        setattr(cfg, split+"_file", file_abs_path)

    cfg.sentencepiece.sentencepiece_model = os.path.join(args.efs, "tokenizer", "mbart50", "bpe", cfg.sentencepiece.sentencepiece_model)
    cfg.vocab_file_src = os.path.join(args.efs, "tokenizer", "mbart50", "dict", cfg.vocab_file_src)
    cfg.vocab_file_tgt = os.path.join(args.efs, "tokenizer", "mbart50", "dict", cfg.vocab_file_tgt)
    cfg.knowledge_file = os.path.join(load_data_dir, "kgtext_wikidata", getattr(cfg, "knowledge_file"))


def update_setting(setting, args):
    for key in setting:
        if getattr(args, key, "#") != "#":
            setting[key] = getattr(args, key)

if __name__ == "__main__":
    
    # set cfg, load config and update cfg
    args = get_dataset_args()
    print(args)
    project_abs_dir = get_abs_project_path()
    load_data_dir = args.load_data_dir
    save_data_dir = args.save_data_dir
    cfg = OmegaConf.load(project_abs_dir + os.sep + "code"+ os.sep + args.config_file)
    setting = OmegaConf.load(project_abs_dir + os.sep + "code"+ os.sep + args.setting_file)
    update_cfg(cfg, args)
    update_setting(setting, args)
    #src_dict = Dictionary.load(cfg.vocab_file_src)
    tgt_dict = Dictionary.load(cfg.vocab_file_tgt)
    bpe = SentencepieceBPE(cfg.sentencepiece)
    

    def token_config_name(setting, args, cfg):
        string = []
        for key, val in setting.items():
            if key == "lang":
                continue
            if val:
                string.append(key)
        return "_".join(string)

    token_style = token_config_name(setting, args, cfg)
    save_data_subdir = os.path.join(save_data_dir, cfg.dataset, setting.lang, args.option, token_style)
    if not os.path.exists(save_data_subdir):
        os.makedirs(save_data_subdir)
    
    for split in ["test", "train", "valid"]:
        data = Kg2KgDataset(cfg, split, tgt_dict, tgt_dict, args)
        save_data_file = os.path.join(save_data_subdir, split)
        if cfg.dataset == "kgtext_wikidata":
            data.write2file2(setting, save_data_file)
        else:
            data.write2file(setting, save_data_file)

