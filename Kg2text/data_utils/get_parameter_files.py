from pickle import LIST
import time
import os
from typing import OrderedDict
from myutils import get_my_args, set_cfg, load_my_cfg
import torch
from kg2textConfig import Kg2textConfig



def yaml_to_sh(file_name):
    now = time.strftime("%Y_%m%d_%H%M", time.localtime(time.time()))
    # set cfg, load config and update cfg
    args = get_my_args()
    print(args)
    cfg = load_my_cfg(args.config)
    #cfg_new = Kg2textConfig()
    #cfg = set_cfg(cfg_new, args, now)

    task_cfg, data_cfg, model_cfg, common, token_cfg = cfg.task, getattr(cfg.task, args.data), cfg.model,cfg.common, cfg.tokenizer
    data_cfg.option, data_cfg.split = task_cfg.option, task_cfg.split
    count = 0
    with open(file_name, "w") as f:
        for idx, key in enumerate(data_cfg):
            if isinstance(data_cfg[key], OrderedDict):
                continue
            count += 1
            f.write("--"+str(key)+" "+str(data_cfg[key])+" ")
            if count == 4:
                f.write(" \\"+"\n")
                count = 0
        f.close()

def args_to_sh(file_name, args):
    assert isinstance(args, list)
    count = 0
    with open(file_name, "w") as f:
        for arg in args:
            count += 1
            f.write(str(arg)+" ")
            if count == 6:
                f.write(" \\"+"\n")
                count = 0
        f.close()

args = [
                "--data",
                "test",
                "--option",
                "train",
                "--split",
                "train",
                "--encoder_type",
                "kgpt",
                "--decoder_type",
                "mbart50_wtags",
                "--tokenizer_src",
                "kgpt_gpt2",
                "--tokenizer_tgt",
                "sentencepiece",
                "--vocab_file_src",
                "dict.gpt2.txt",
                "--vocab_file_tgt",
                "dict.mbart50_wtags.txt",
                "--experiment_comment",
                "kgpt_mbart50_added_tags_tgt_lang_copy_gate",
                "--pretrained_encoder_file",
                "kgpt_mbart50_encoder.pt",
                "--pretrained_decoder_file",
                "mbart50_mbart50_decoder_wtags.pt",
                "--prepend_src_lang_tag",
                "False",
                "--prepend_tgt_lang_tag",
                "True",
                "--src_length",
                "256",
                "--tgt_length",
                "108",
                "--seed",
                "0"
            ]




if __name__ == '__main__':
    yaml_to_sh("./code/data_parameters_from_yaml.sh")
    #args_to_sh("./code/data_parameters_from_json.sh", args)
    
    


    



args = [
                "--data",
                "test",
                "--option",
                "train",
                "--split",
                "train",
                "--encoder_type",
                "kgpt",
                "--decoder_type",
                "mbart50_wtags",
                "--tokenizer_src",
                "kgpt_gpt2",
                "--tokenizer_tgt",
                "sentencepiece",
                "--vocab_file_src",
                "dict.gpt2.txt",
                "--vocab_file_tgt",
                "dict.mbart50_wtags.txt",
                "--experiment_comment",
                "kgpt_mbart50_added_tags_tgt_lang_copy_gate",
                "--pretrained_encoder_file",
                "kgpt_mbart50_encoder.pt",
                "--pretrained_decoder_file",
                "mbart50_mbart50_decoder_wtags.pt",
                "--prepend_src_lang_tag",
                "False",
                "--prepend_tgt_lang_tag",
                "True",
                "--src_length",
                "256",
                "--tgt_length",
                "108",
                "--seed",
                "0"
            ]

