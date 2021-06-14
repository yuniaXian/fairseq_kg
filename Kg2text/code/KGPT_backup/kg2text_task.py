from fairseq.tasks.fairseq_task import FairseqTask
from myutils import *
from omegaconf import DictConfig
from kg2textConfig import *
import os.path as op
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from DataLoader import WikiDataDataset, WebNLGDataset, WikiBioNLGDataset
# gpt2 dict: https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
#
# task: dict, tokenizer_type, cfg.model -> build model

#@register_task("kg2text_task", dataclass=Kg2textTask)
class Kg2textTask(FairseqTask):
    cfg: Kg2textTaskConfig

    def __init__(self, cfg: Kg2textConfig, args, src_dict, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.data_cfg = getattr(cfg.task, args.data, "webnlg")
        # self.token_cfg = cfg.tokenizer
        self.model_cfg = cfg.model

        """
        if args.data == "webnlg":
            self.data_cfg = cfg.webnlg
        if args.data == "wikidata":
            self.data_cfg = cfg.wikidata
        if args.data == "wikibionlg":
            self.data_cfg = cfg.wikibionlg
        self.token_cfg = cfg.token
        # self.seed
        """

    @classmethod
    def setup_task(cls, cfg: Kg2textConfig, args):
        data_cfg = getattr(cfg.task, args.data, "webnlg")
        #token_cfg = cfg.task

        dict_add_tags_src = args.dict_add_tags_src
        dict_add_tags_tgt = args.dict_add_tags_tgt
        ent = "[ENT]"
        pred = "[PRED]"
        sub = "[SUB]"
        triple = "[TRIPLE]"
        tags = [ent, pred, sub, triple]
        if not op.isfile(data_cfg.vocab_file_src):
            raise FileNotFoundError(f"Dict not found: {data_cfg.vocab_file_src}")

        if dict_add_tags_src:
            src_dict = load_dictionary(data_cfg, "src", tags)
        else:
            src_dict = load_dictionary(data_cfg, "src")
        logger.info(
            f"source dictionary size ({data_cfg.vocab_file_src}): " f"{len(src_dict):,}"
        )
        if data_cfg.tokenizer_src == data_cfg.tokenizer_tgt:
            tgt_dict = src_dict
            logger.info(f"target dictionary size ({data_cfg.vocab_file_tgt}): " f"{len(tgt_dict):,}")
            return cls(cfg, args, src_dict, tgt_dict)

        if not op.isfile(data_cfg.vocab_file_tgt):
            raise FileNotFoundError(f"Dict not found: {data_cfg.vocab_file_tgt}")
        if dict_add_tags_tgt:
            tgt_dict = load_dictionary(data_cfg, "tgt", tags)
        else:
            tgt_dict = load_dictionary(data_cfg, "tgt")
        logger.info(
            f"target dictionary size ({data_cfg.vocab_file_tgt}): " f"{len(tgt_dict):,}"
        )
        """
        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        """
        return cls(cfg, args, src_dict, tgt_dict)

    @property
    def target_dictionary(self):
        return self.tgt_dict
    @property
    def source_dictionary(self):
        return self.src_dict
    @property
    def max_positions(self):
        """
        return self.cfg.max_source_positions, self.cfg.max_target_positions
        """
        # TODO cfg.max_source_positions, cfg.max_target_positions
        return 1024

    def build_model(self, cfg):

        pass

    def build_bpe(self, cfg):
        """
        token_cfg = cfg.token
        logger.info(f"tokenizer: {self.token_cfg.bpe_tokenizer_src}")
        return encoders.build_bpe(Namespace(**self.token_cfg.bpe_tokenizer_src))
        """
        pass

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        data_cfg: FairseqDataclass = None,
        **kwargs
    ):
        #data_cfg = getattr(task_cfg, split)
        """
        cfg.train_file = get_kg2text_abs_path("data", cfg, cfg.train_file)
        cfg.eval_file = get_kg2text_abs_path("data", cfg, cfg.eval_file)
        cfg.test_file = get_kg2text_abs_path("data", cfg, cfg.test_file)
        """

        file_path = getattr(self.data_cfg, split+"_file")

        if data_cfg.dataset == 'wikidata':
            self.data = WikiDataDataset(data_cfg, self.src_dict, self.tgt_dict)
        elif data_cfg.dataset == "webnlg":
            self.data = WebNLGDataset(data_cfg, self.src_dict, self.tgt_dict)
        elif data_cfg.dataset == "wikibionlg":
            self.data = WikiBioNLGDataset(data_cfg, self.src_dict, self.tgt_dict)
        else:
            raise NotImplementedError("This dataset is not yet supported")



        return self.data

    def dataset(self, split):
        if self.datasets[split]:
            return self.datasets[split]
        else:
            raise FileNotFoundError




class mytokenizer:

    def __init__(self, bpe, dict):
        self.bpe, self.dict = bpe, dict

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

    def decode(self, tensor, prepend_lang_tag=False, lang=None):
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

"""
    print(torch.cuda.is_available())
    
    args = get_my_args()
    mycfg = Kg2textConfig()
    cfg_in = load_my_cfg(args.config)
    mycfg = myConfigUpdate(cfg_in, mycfg)
    cfg = mycfg.task.test
    cfg.vocab_file_src = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_src)
    cfg.vocab_file_tgt = get_kg2text_abs_path("dict", cfg, cfg.vocab_file_tgt)
    tokenizer = mytokenizer.build_tokenizer(mycfg.task.test, "tgt")
    
    text = "windy"
    #print(tokenize_text_test(mycfg.task.test, text))
    t = tokenizer.encode(text)
    
    s = tokenizer.bpe.bpe.bpe("greenhand")
    
    
    print("Done")
    
    eval_data = WebNLGDataset(mycfg.task.test)
    
    
    task = Kg2textTask.setup_task(mycfg, args)
    for step, batch in enumerate(eval_data):
        print("1")
    from kg2text_model import Kg2textTransformerModel
    model = Kg2textTransformerModel.build_model(mycfg.model, task)
    
    mycfg.model.checkpoint_file = get_kg2text_abs_path("model_kgpt", mycfg.model, mycfg.model.checkpoint_file)
    
    reloaded = torch.load(mycfg.model.checkpoint_file, map_location=torch.device('cpu'))
    
    printMyModel(model)
    saveMyModel(model, "encoder", mycfg.model)
    saveMyModel(model, "decoder", mycfg.model)

"""