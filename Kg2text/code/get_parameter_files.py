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
args=[
                "Kg2text/data-bin/webnlg/sentencepiece_bped_data",
                "--encoder-normalize-before", "--decoder-normalize-before",
                "--arch", "mbart_large", "--task", "multilingual_denoising",
                "--criterion", "label_smoothed_cross_entropy", "--label-smoothing", "0.2",
                "--dataset-impl", "raw", "--optimizer", "adam", "--adam-eps", "1e-06", "--adam-betas", "(0.9, 0.98)",
                "--lr-scheduler", "polynomial_decay", "--lr", "3e-05", "--stop-min-lr", "-1",
                "--warmup-updates", "2500", "--max-update", "40000", "--total-num-update", "40000",
                "--dropout", "0.3", "--attention-dropout", "0.1",
                "--weight-decay", "0.0", "--max-tokens", "1024", "--update-freq", "2", "--save-interval", "1",
                "--save-interval-updates", "8000", "--keep-interval-updates", "10", "--no-epoch-checkpoints",
                "--seed", "222", "--log-format", "simple", "--log-interval", "2",
                "--reset-optimizer", "--reset-meters", "--reset-dataloader", "--reset-lr-scheduler",
                "--save-dir", "checkpoint/denoising_webnlg",
                "--layernorm-embedding", "--ddp-backend", "no_c10d",
                "--langs", "en_XX",
                "--no-whole-word-mask-langs", "False",
                "--tokens-per-sample", "512",
                "--sample-break-mode", "eos",
                "--mask", "0.5",
                "--mask-random", "0.0",
                "--insert", "0.0",
                "--permute", "0.0",
                "--rotate", "0.5",
                "--poisson-lambda", "3.0",
                "--permute-sentences", "0.0",
                "--mask-length", "word",
                "--replace-length", "-1",
                "--shorten-method", "none",
                "--bpe", "sentencepiece",
                "--sentencepiece-model", "Kg2text/tokenizer/mbart50/bpe/sentence.bpe.model",
                "--train-subset", "train.input",
                "--valid-subset", "eval.input"
            ]




if __name__ == '__main__':
    #yaml_to_sh("./code/data_parameters_from_yaml.sh")
    args_to_sh("./Kg2text/code/denoising_task.sh", args)
    
    


    

