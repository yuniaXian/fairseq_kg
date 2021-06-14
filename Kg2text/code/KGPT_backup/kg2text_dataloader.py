from fairseq.data.fairseq_dataset import FairseqDataset
from kg2textConfig import *
from myutils import *
import json
from fairseq.data.data_utils import load_indexed_dataset
import random

#dict = Dictionary.load("/media/MyDataStor1/jxian/efs-storage/Kg2text/dataset/dict/dict.gpt2.txt")
#data = load_indexed_dataset("/media/MyDataStor1/jxian/efs-storage/Kg2text/dataset/webnlg/test.txt", dictionary=dict)

#print("DFD")

def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return


class Kg2textBaseDataset(FairseqDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, dataset):
        super(Kg2textBaseDataset, self).__init__()

        # cfg = cfg.cfg
        assert isinstance(cfg, Kg2textDataSetConfig)
        # TODO: add all needed para
        # self.token_cfg = cfg.token_cfg
        # for Kgpt_gpt2
        cfg.vocab_file_src = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_src)
        cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)

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

        self.max_entity = cfg.max_entity
        self.max_fact = cfg.max_fact
        self.forbid_duplicate_relation = cfg.forbid_duplicate_relation
        self.percent = cfg.percent
        self.lower_case = cfg.lower_case

        self.src_tokenizer = cfg.tokenizer_src  #
        self.src_bpe = load_bpe_tokenizer(cfg.tokenizer_src, cfg)
        self.src_dict = Dictionary.load(cfg.vocab_file_src)
        self.src_max = cfg.src_length
        self.src_lang_tag_template = cfg.src_lang_tag_template
        self.src_enc_type = cfg.encoder_type
        self.src_lang = cfg.src_lang
        self.prepend_src_lang_tag = cfg.prepend_src_lang_tag

        self.tgt_tokenizer = cfg.tokenizer_tgt
        self.tgt_bpe = load_bpe_tokenizer(cfg.tokenizer_tgt, cfg)
        self.tgt_dict = Dictionary.load(cfg.vocab_file_tgt)
        self.tgt_max = cfg.tgt_length
        self.tgt_lang_tag_template = cfg.tgt_lang_tag_template
        self.tgt_enc_type = cfg.tokenizer_tgt
        self.tgt_lang = cfg.tgt_lang
        self.prepend_tgt_lang_tag = cfg.prepend_tgt_lang_tag

        # self.data
        cfg.train_file = get_kg2text_abs_path("data", cfg, cfg.train_file)
        cfg.eval_file = get_kg2text_abs_path("data", cfg, cfg.eval_file)
        cfg.test_file = get_kg2text_abs_path("data", cfg, cfg.test_file)

        self.data = dataset

    def tokenize_text_kgpt(self, text):
        pass

    def tokenize_text(self, text, source: str = "tgt", lang=None):
        # GPT2Tokenizer: add all tags -> tokenize
        # fairseq_gpt2 / sentencepiece bpe -> add tag -> dictionary encode
        # todo don't check each time
        if source == "src":
            bpe_tokenizer = self.src_bpe
            lang_tag_template = self.src_lang_tag_template
            prepend_lang_tag = self.prepend_src_lang_tag
            token_dict = self.src_dict
            tokenizer_name = self.src_tokenizer
        elif source == "tgt":
            bpe_tokenizer = self.tgt_bpe
            lang_tag_template = self.tgt_lang_tag_template
            prepend_lang_tag = self.prepend_tgt_lang_tag
            token_dict = self.tgt_dict
            tokenizer_name = self.tgt_tokenizer
        else:
            print("func: tokenize_text: source if not specified for bpe and dict")
        # text '[ENT] Sweet potato' ->  [50257, 36087, 21219] /
        # '▁[ ENT ] ▁Sweet ▁po tato' -> "[EN_XX] ▁Sweet ▁po tato" -> tensor([ 3, 378, 20157, 268, 87497, 160, 28647, 2])
        if tokenizer_name == "kgpt_gpt2":
            if prepend_lang_tag and lang:
                text = self.lang_tagging(source, text)
            text_done = bpe_tokenizer.encode(text, add_special_tokens=False)
            return text_done
        else:
            text_tokenized = bpe_tokenizer(text)
            if lang is not None and prepend_lang_tag:
                # assume the tgt_dict already have language tags in the dict, e.g. mBART50 task loaded dictionary
                # so prepend the tag direct and have encode_line convert it to ID.
                text_tokenized = self.lang_tagging(source, text_tokenized)
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

    def linearize_v2(self, entity, lang=None, lower_case=False):
        """
        get all the information in knowledge-full.json for entity
        return string_tokenized, triple_id
        """
        # entity: ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
        if lower_case:
            entity0, entity1 = entity[0].lower(), entity[1].lower()
        else:
            entity0, entity1 = entity[0], entity[1]

        if self.prepend_src_lang_tag:
            entity_tagged = self.lang_tagging(entity0)
        else:
            entity_tagged = "[ENT] {}".format(entity0)
        des_tagged = "[TRIPLE] [PRED] description [SUB] {} [TRIPLE]".format(entity1)
        entity_tokenized = self.tokenize_text(entity_tagged, "src", None)
        des_tokenized = self.tokenize_text(des_tagged, "src", None)
        string = entity_tokenized + des_tokenized
        triple_id = [1] * len(entity_tokenized) + [2] * len(des_tokenized)

        added = set()
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if lower_case:
                    rel_tagged = "[PRED] {} [SUB] {} [TRIPLE]".format(rel[0].lower(), rel[1].lower())
                else:
                    rel_tagged = "[PRED] {} [SUB] {} [TRIPLE]".format(rel[0], rel[1])
                rel_tokenized = self.tokenize_text(rel_tagged, "src", None)
                string += rel_tokenized
                triple_id += [triple_id[-1] + 1] * len(rel_tokenized)
                added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        return string, triple_id

    def sentence_tagging(self, sentence, prepend_lang_tag):
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
        bos = self.tgt_bpe.bos_token
        eos = self.tgt_bpe.eos_token
        if self.prepend_tgt_lang_tag:
            sent_format = self.tgt_lang_tag_template.format(self.tgt_lang) + " {} " + eos
        else:
            sent_format = "[SOS] {} [EOS]"
        sent_tagged = sent_format.format(sentence)

        return sent_tagged

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

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class Kg2textDownStreamDataset(Kg2textBaseDataset):
    def __init__(self, cfg: Kg2textDataSetConfig, dataset):
        super(Kg2textDownStreamDataset, self).__init__(cfg, dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        random.seed(0)
        sentence = random.choice(entry['text'])
        KBs = entry['kbs']  # set of triples

        # for src_text
        if self.src_enc_type == 'sequence':
            strings = []
            entity_ids = []
            triple_ids = []

            for i, entity_label in enumerate(KBs):  # entity_label: "W1215"
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label]  # ['Sweet potato', 'Sweet potato', [['main ingredients', 'Binignit']]]
                string, triple_id = self.linearize_v2(entity)
                strings += string
                entity_ids += [i + 1] * len(string)
                triple_ids += triple_id

            position_ids = list(range(len(strings)))
            assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

            input_ids, entity_ids, triple_ids, position_ids = self.entity_truncated(
                strings, entity_ids, triple_ids, position_ids)

            sent_tagged = self.sentence_tagging(sentence, self.prepend_tgt_lang_tag)
            sent_tokenized = self.tokenize_text(sent_tagged, "tgt", None)
            output_ids = self.sentence_truncated(sent_tokenized)

            return input_ids, entity_ids, triple_ids, position_ids, output_ids[:-1], output_ids[1:]

        else:
            raise NotImplementedError

    def get_batch_shapes(self):
        shape_src = (self.src_max, self.batch_size)
        # shape_tgt = (self.tgt_max, self.batch_size)

        return shape_src

    def num_tokens(self, index):
        return self.src_max

    def num_tokens_vec(self, indices):
        return [self.src_max] * len(indices)
