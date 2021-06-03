from math import log, log10
import torch
import json
import random
from torch.utils.data import Dataset
from fairseq.data import Dictionary
from kg2textConfig import Kg2textConfig, Kg2textTokenConfig, Kg2textDataSetConfig, Kg2textTaskConfig
import logging
import os.path as op
from myutils import load_bpe_tokenizer

random.seed(0)
logger = logging.getLogger(__name__)

# cfg, tasl_cfg, model_cfg, token_cfg

def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return

class KBDataset(Dataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(KBDataset, self).__init__()

        #cfg = cfg.cfg
        assert isinstance(cfg, Kg2textDataSetConfig)
        # TODO: add all needed para
        # self.token_cfg = cfg.token_cfg
        # for Kgpt_gpt2
        ent = "[ENT]"
        pred = "[PRED]"
        sub = "[SUB]"
        triple = "[TRIPLE]"
        tags = [ent, pred, sub, triple]
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_bpe = load_bpe_tokenizer(cfg.tokenizer_src, cfg)
        self.tgt_bpe = load_bpe_tokenizer(cfg.tokenizer_tgt, cfg)

        self.src_lang_tag = cfg.src_lang_tag_template.format(cfg.src_lang)
        self.tgt_lang_tag = cfg.tgt_lang_tag_template.format(cfg.tgt_lang)
        self.lang_entity_format = cfg.src_lang_tag_template.format(cfg.src_lang) + " {}"
        self.entity_format = ent + " {}"
        # words_tagged = '[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower())
        self.des_format = triple + " " + pred + " description " + sub + " {} " + triple
        self.rel_format = pred + " {} " + sub + " {} " + triple

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

        self.src_tokenizer = cfg.tokenizer_src #
        self.src_max = cfg.src_length
        self.src_lang_tag_template = cfg.src_lang_tag_template
        self.src_enc_type = cfg.encoder_arch
        self.src_lang = cfg.src_lang
        self.prepend_src_lang_tag = cfg.prepend_src_lang_tag

        self.tgt_tokenizer = cfg.tokenizer_tgt
        self.tgt_max = cfg.tgt_length
        self.tgt_lang_tag_template = cfg.tgt_lang_tag_template
        self.tgt_lang = cfg.tgt_lang
        self.prepend_tgt_lang_tag = cfg.prepend_tgt_lang_tag


        file_path = getattr(cfg, cfg.split+"_file", "")
        if file_path == "":
            logger.warning("Dataset for task: ", cfg.option, " is not specified, using eval dataset.")
            file_path = cfg.eval_file
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            print("Loaded data from ", file_path, " for task ", cfg.split)

        if cfg.percent > 1:
            self.data = self.data[:int(self.percent)]
        else:
            selected_size = int(len(self.data) * self.percent)
            self.data = self.data[:selected_size]

    def tokenize_text(self, text, tokenizer, type):
        if type in ["sentencepiece", "mbart", "mbart50"]:
            tokenized_text = tokenizer.encode_line(text)
        elif type in ["kgpt", "gpt2"]:
            tokenized_text = tokenizer.encode(text)
        else:
            raise NotImplementedError
        return tokenized_text
            
    def tokenize_text0(self, text, source: str = "tgt", lang=None):
        # GPT2Tokenizer: add all tags -> tokenize
        # fairseq_gpt2 / sentencepiece bpe -> add tag -> dictionary encode
        # todo don't check each time
        if source == "src":
            bpe_tokenizer = self.src_bpe
            token_dict = self.src_dict
            tokenizer_name = self.src_tokenizer
        elif source == "tgt":
            bpe_tokenizer = self.tgt_bpe
            token_dict = self.tgt_dict
            tokenizer_name = self.tgt_tokenizer
        else:
            print("func: tokenize_text: source if not specified for bpe and dict")
        # text '[ENT] Sweet potato' ->  [50257, 36087, 21219] /
        # '▁[ ENT ] ▁Sweet ▁po tato' -> "[EN_XX] ▁Sweet ▁po tato" -> tensor([ 3, 378, 20157, 268, 87497, 160, 28647, 2])
        if tokenizer_name == "kgpt_gpt2":

            text_done = bpe_tokenizer.encode(text, add_special_tokens=False)
            return text_done
        else:
            text_tokenized = bpe_tokenizer(text)

            text_done = token_dict.encode_line(text_tokenized, add_if_not_exist=False, append_eos=False)

            return text_done

    def lang_tagging(self, text):
        # prepend lang_tag after bpe encoding
        text_tagged = "{lang_tag} {tokenized}".format(
            lang_tag=self.src_lang_tag_template.format(self.src_lang),
            tokenized=text)
        """
        if source == "src":
            text_tagged = "{lang_tag} {tokenized}".format(
                lang_tag=self.src_lang_tag_template.format(self.src_lang),
                tokenized=text)

        elif source == "tgt":
            text_tagged = "{lang_tag} {tokenized}".format(
                lang_tag=self.tgt_lang_tag_template.format(self.tgt_lang),
                tokenized=text)
        else:
            raise NotImplementedError
        """
        return text_tagged

    def linearize_v1(self, entity, lower_case=False, lang=None):

        if lower_case:
            entity0, entity1 = entity[0].lower(), entity[1].lower()
        else:
            entity0, entity1 = entity[0], entity[1]

        entity0, entity1 = self.src_bpe.encode(entity0), self.src_bpe.encode(entity1)

        if self.prepend_src_lang_tag:
            entity_tagged = self.lang_tagging(entity0)
        else:
            entity_tagged = self.entity_format.format(entity0)

        des_tagged = self.des_format.format(entity1)

        entity_tokenized = self.src_dict.encode_line(entity_tagged, add_if_not_exist=False, append_eos=False)
        des_tokenized = self.src_dict.encode_line(des_tagged, add_if_not_exist=False, append_eos=False)
        string = entity_tokenized.tolist() + des_tokenized.tolist()
        #string = torch.cat((entity_tokenized, des_tokenized), 0)
        triple_id = [1] * len(entity_tokenized) + [2] * len(des_tokenized)
        #triple_id = [1] * entity_tokenized.shape[0] + [2] * des_tokenized.shape[0]
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
                rel_tokenized = self.src_dict.encode_line(rel_tagged, add_if_not_exist=False, append_eos=False)
                #string = torch.cat((string, rel_tokenized), 0)
                string += rel_tokenized.tolist()
                triple_id += [triple_id[-1] + 1] * len(rel_tokenized)
                added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        return string, triple_id

    def linearize_v2(self, entity, lang=None, lower_case=False):
        """
        get all the information in knowledge-full.json for entity
        return string_tokenized, triple_id
        """
        # entity: ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
        """
        ent = "[ENT]"
        pred = "[PRED]"
        sub = "[SUB]"
        triple = "[TRIPLE]"
        self.src_lang_tag = cfg.src_lang_tag_template.format(cfg.src_lang)
        self.lang_entity_format = cfg.src_lang_tag_template.format(cfg.src_lang) + " {}"
        self.entity_format = ent + " {}"
        # words_tagged = '[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower())
        self.des_format = triple + " " + pred + " description " + sub + " {} " + triple
        self.rel_format = pred + " {} " + sub + " {} " + triple

        # words_tagged = '[en_XX] {}'.format(entity[0].lower())
        src_lang_tag = self.src_lang_tag_template.format(self.src_lang)
        # words_tagged = '[ENT] {}'.format(entity[0].lower())
        lang_entity_format = src_lang_tag + " {}"
        entity_format = ent + " {}"
        #words_tagged = '[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower())
        des_format = triple + " " + pred + " description " +sub + " {} " + triple
        rel_format = pred + " {} " + sub + " {} " + triple
        """
        if lower_case:
            entity0, entity1 = entity[0].lower(), entity[1].lower()
        else:
            entity0, entity1 = entity[0], entity[1]

        if self.prepend_src_lang_tag:
            entity_tagged = self.lang_tagging(entity0)
        else:
            entity_tagged = self.entity_format.format(entity0)

        des_tagged = self.des_format.format(entity1)

        entity_tokenized = self.src_bpe.encode(entity_tagged)
        des_tokenized = self.src_bpe.encode(des_tagged)

        string = entity_tokenized + des_tokenized
        triple_id = [1] * len(entity_tokenized) + [2] * len(des_tokenized)

        added = set()
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if lower_case:
                    rel_tagged = self.rel_format.format(rel[0].lower(), rel[1].lower())
                else:
                    rel_tagged = self.rel_format.format(rel[0], rel[1])
                rel_tokenized = self.src_bpe.encode(rel_tagged)
                string += rel_tokenized
                triple_id += [triple_id[-1] + 1] * len(rel_tokenized)
                added.add(rel[0])
            
            if len(added) >= self.max_fact:
                break
        
        return string, triple_id

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

    def linearize(self, entity, lower_case=False, lang=None):
        if "gpt" in self.src_tokenizer:
            string, triple_id = self.linearize_v2(entity, self.lower_case)

        elif self.src_tokenizer in ["mbart", "sentencepiece", "mbart50", "mbart50t"]:
            string, triple_id = self.linearize_v1(entity, self.lower_case)
        else:
            raise NotImplementedError

        return string, triple_id

    def sentence_preprocess_v2(self, sentence, prepend_lang_tag, lower_case=False):
        if lower_case:
            sentence =sentence.lower()

        if self.prepend_tgt_lang_tag:
            sent_format = self.lang_text_format
        else:
            sent_format = self.text_format
        sent_tagged = sent_format.format(sentence)
        sent_tokenized = self.tgt_bpe.encode(sent_tagged)
        return sent_tokenized

    def sentence_preprocess_v1(self, sentence, prepend_lang_tag, lower_case=False, add_bos=True, add_eos=False):
        # GPT2Tokenizer: add all tags -> tokenize
        # fairseq_gpt2 / sentencepiece bpe -> add tag -> dictionary encode
        # TODO
        """
        if self.tgt_tokenizer in ["fairseq_gpt2", "mbart50", "sentencepiece"]:
            bos = self.tgt_dict.bos_word
            eos = self.tgt_dict.eos_word
            #pad_index = self.tgt_dict.pad_index
        elif self.tgt_tokenizer == "kgpt_gpt2":
            bos = self.tgt_bpe.bos_token
            eos = self.tgt_bpe.eos_token
            #pad_index = self.tgt_bpe.pad_token_id
        else:
            raise NotImplementedError
        """
        if lower_case:
            sentence =sentence.lower()
        sent_bped = self.tgt_bpe.encode(sentence)
        if self.prepend_tgt_lang_tag:
            sent_format = self.lang_text_format
        else:
            sent_format = self.text_format
        sent_tagged = sent_format.format(sent_bped)
        sent_tokenized = self.tgt_dict.encode_line(sent_tagged).tolist()
        

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

    def sentence_preprocess(self, sentence, prepend_lang_tag, lower_case=False):
        if "gpt" in self.tgt_tokenizer:
            sent_tokenized = self.sentence_preprocess_v2(sentence, self.prepend_tgt_lang_tag)
        # string: all the knowledge(description, rels) tokenized vector
        # triple_id: indicate different triples in the string
        elif self.tgt_tokenizer in ["mbart", "sentencepiece", "mbart50"]:
            sent_tokenized = self.sentence_preprocess_v1(sentence, self.prepend_tgt_lang_tag)
        else:
            raise NotImplementedError
        return sent_tokenized

    def sentence_truncated(self, sentence):
        if self.tgt_tokenizer in ["fairseq_gpt2", "mbart50", "sentencepiece"]:
            pad_index = self.tgt_dict.pad_index
        elif self.tgt_tokenizer == "kgpt_gpt2":
            pad_index = self.tgt_bpe.pad_token_id
        else:
            raise NotImplementedError

        if len(sentence) >= self.tgt_max:
            sentence_ids = torch.LongTensor(sentence[:self.tgt_max])
        else:
            sentence_ids = torch.LongTensor(
                sentence[:self.tgt_max] + [pad_index] * (self.tgt_max - len(sentence)))

        return sentence_ids

    def entity_truncated(self, strings, entity_ids, triple_ids, position_ids):
        if self.src_tokenizer in ["fairseq_gpt2", "mbart50", "sentencepiece"]:
            pad_index = self.src_dict.pad_index
        elif self.src_tokenizer == "kgpt_gpt2":
            pad_index = self.src_bpe.pad_token_id
        else:
            raise NotImplementedError

        if len(strings) >= self.src_max:
            input_ids = torch.LongTensor(strings[:self.src_max])
            entity_ids = torch.LongTensor(entity_ids[:self.src_max])
            triple_ids = torch.LongTensor(triple_ids[:self.src_max])
            position_ids = torch.LongTensor(position_ids[:self.src_max])
        else:
            input_ids = torch.LongTensor(strings + [pad_index] * (self.src_max - len(strings)))
            entity_ids = torch.LongTensor(entity_ids + [0] * (self.src_max - len(strings)))
            triple_ids = torch.LongTensor(triple_ids + [0] * (self.src_max - len(strings)))
            position_ids = torch.LongTensor(position_ids + [0] * (self.src_max - len(strings)))

        return input_ids, entity_ids, triple_ids, position_ids
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError

class WikiDataDataset(KBDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(WikiDataDataset, self).__init__(cfg, src_dict, tgt_dict)

        with open(cfg.knowledge_path, 'r') as f:
            self.knowledge = json.load(f)
        print("Total samples = {}; Total entities = {}".format(len(self.data), len(self.knowledge)))

    def get_reference(self, idx, lower_case=False):
        if self.lower_case: # all words in 'text' part -> lower case
            return [[_.lower() for _ in self.data[idx]['text']]]
        else:
            return [self.data[idx]['text']]  #

    def get_entities(self, idx):
        entry = self.data[idx]
        entities = []
        for _ in entry['kblinks']:
            if _ is not None and _ in self.knowledge and _ not in entities:
                entities.append(_)
        if 'title' in entry:
            entities.insert(0, "TITLE:::" + entry['title_kb_id'])
        return entities

    def __getitem__(self, idx):
        entry = self.data[idx]

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

class DownStreamDataset(KBDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(DownStreamDataset, self).__init__(cfg, src_dict, tgt_dict)

        print("Total samples = {}".format(len(self.data)))

    def get_reference(self, idx, do_lower_case=False):
        if do_lower_case:
            return [_.lower().split(' ') for _ in self.data[idx]['text']]
        else:
            return [_.split(' ') for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text']) # TODO
        KBs = entry['kbs'] # set of triples

        # for src_text
        if self.src_enc_type == 'sequence':
            strings = []
            entity_ids = []
            triple_ids = []

            for i, entity_label in enumerate(KBs): # entity_label: "W1215"
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label] # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                string, triple_id = self.linearize(entity)
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

class WebNLGDataset(DownStreamDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(WebNLGDataset, self).__init__(cfg, src_dict, tgt_dict)
        
class WebNLGChallengeDataset(DownStreamDataset):
    def __init____init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(WebNLGChallengeDataset, self).__init__(cfg, src_dict, tgt_dict)
        
class E2ENLGDataset(DownStreamDataset):
    def __init__(self, cfg: Kg2textDataSetConfig):
        super(E2ENLGDataset, self).__init__(cfg)

class LogicNLGDataset(DownStreamDataset):
    def __init__(self, cfg: Kg2textDataSetConfig):
        super(LogicNLGDataset, self).__init__(cfg)
        
class WikiBioNLGDataset(DownStreamDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict):
        super(WikiBioNLGDataset, self).__init__(cfg, src_dict, tgt_dict)
        
class GPTDataset(KBDataset):
    def __init__(self, cfg: Kg2textDataSetConfig):
        super(GPTDataset, self).__init__(cfg)

        print("Total samples = {}".format(len(self.data)))
    
    def get_reference(self, idx):
        return [_.split(' ') for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text'])

        KBs = entry['kbs']

        strings = ''
        for i, entity_label in enumerate(KBs):
            entity = KBs[entity_label]

            name = entity[0]
            
            for rel in entity[-1]:
                strings += ' {} {} {} . '.format(name, rel[0], rel[1])

        KB_ids = self.src_tokenizer.encode(strings, add_special_tokens=False)
        
        if len(KB_ids) < self.src_max:
            KB_ids = [self.src_dict.pad_index] * (self.src_max - len(KB_ids)) + KB_ids
        else:
            KB_ids = KB_ids[:self.src_max]

        inputs = torch.LongTensor(KB_ids)

        symbolized_format = self.tgt_dict.bos + " {} " + self.tgt_dict.eos
        sentence = self.tgt_tokenizer.encode(symbolized_format.format(sentence), add_special_tokens=False)
        if len(sentence) >= self.tgt_max:
            output_ids = torch.LongTensor(sentence[:self.tgt_max])
        else:
            output_ids = torch.LongTensor(sentence[:self.tgt_max] + [self.tgt_dict.pad_index] * (self.tgt_max - len(sentence)))

        return inputs, output_ids[:-1], output_ids[1:]

class FairseqTranferDataset(KBDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, src_dict, tgt_dict, add_bos=True, add_eos=False):
        super(FairseqTranferDataset, self).__init__(cfg, src_dict, tgt_dict)
        self.add_bos = add_bos
        self.add_eos = add_eos
        print("Total samples = {}".format(len(self.data)))

    def get_reference(self, idx, do_lower_case=False):
        if do_lower_case:
            return [_.lower().split(' ') for _ in self.data[idx]['text']]
        else:
            return [_.split(' ') for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text'])
        KBs = entry['kbs'] # set of triples

        # for src_text
        if self.src_enc_type == 'sequence':
            strings = ""
            for i, entity_label in enumerate(KBs): # entity_label: "W1215"
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label] # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                string = self.linearize_bpe(entity)
                strings += " " + string

            if self.prepend_src_lang_tag:
                strings = self.lang_tagging(strings[1:])
            sent_tagged = self.sentence_preprocess_bpe(sentence, self.prepend_tgt_lang_tag, self.add_bos, self.add_eos)

            return {"text_bped": strings, "sent_bped": sent_tagged}

        else:
            raise NotImplementedError



