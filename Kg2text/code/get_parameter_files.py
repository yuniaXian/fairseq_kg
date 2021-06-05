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

def sh_to_args(sh_file, to_write):
    args = []
    f = open(sh_file, "r")
    lines = f.readlines()
    f.close()
    
    for line in lines:
        line = line.split()
        for para in line: # the last symbol is "\\"
            if para == "\\":
                continue
            para.replace("\"", "")
            para.replace("\'", "")
            args.append("\""+para+"\"")
            #args.append(para.replace("\'", "\""))
    print(args)
    with open(to_write, "w") as fw:
        fw.write("args=["+"\n")
        count = 0
        for arg in args:
            fw.write(arg+", ")
            count += 1
            if count >=6:
                fw.write("\n")
                count = 0
        fw.write("\n")
        fw.write("]")
    fw.close()


        


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


KG2TEXT = "/home/xianjiay/efs-storage/workspaces/hoverboard/fairseq/Kg2text"
EXPERIMENT = "//home/xianjiay/efs-storage/workspaces/hoverboard/fairseq/Kg2text/experiment"
if __name__ == '__main__':
    #yaml_to_sh("./code/data_parameters_from_yaml.sh")
    args=[
        "--option", "kg2kg",
        "--seperate", "",
        "--text_only", "",
        "--simple", "",
        "--tokenized", "True",
        "--tagged", "True", 
        "--dataset", "kgtext_wikidata",
        "--add_eos", "",
        "--add_bos", "",
        "--lang", "en_XX",
        "--config_file", "triples_dataset.yaml",
        "--setting_file", "token_setting.yaml",
        "--load_data_dir", "/home/xianjiay/efs-storage/data-bin/dataset",
        "--save_data_dir", "/home/xianjiay/efs-storage/data-bin/dataset_denoising",
        "--efs", "/home/xianjiay/efs-storage"
      ]


    args_to_sh("./Kg2text/code/create_dataset_kg_text_wikidata.sh", args)
    #sh_to_args(EXPERIMENT + "/translation_task_args_from_sh.sh", EXPERIMENT+"/translation_task_args_from_sh.txt")
    
    


    

