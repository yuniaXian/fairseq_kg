from myutils import get_kg2text_abs_path, get_my_args, set_cfg
import time
import torch
from kg2textConfig import Kg2textConfig
from kg2text_task import Kg2textTask

def build_fairseq_dataset(cfg, split, src_dict, tgt_dict, add_bos=True, add_eos=False):

    cfg.split = split
    from DataLoader import FairseqTranferDataset
    data = FairseqTranferDataset(cfg, src_dict, tgt_dict, add_bos, add_eos)
    from torch.utils.data import DataLoader
    #dataloader = DataLoader(data)
    L = len(data)
    input_file = split + ".input"
    label_file = split + ".label"
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

if __name__ == "__main__":
    now = time.strftime("%Y_%m%d_%H%M", time.localtime(time.time()))
    # set cfg, load config and update cfg
    args = get_my_args()
    print(args)
    cfg_new = Kg2textConfig()
    cfg = set_cfg(cfg_new, args, now)

    task_cfg, data_cfg, model_cfg, common, token_cfg = cfg.task, getattr(cfg.task, args.data), cfg.model,cfg.common, cfg.tokenizer
    data_cfg.option, data_cfg.split = task_cfg.option, task_cfg.split
    task = Kg2textTask.setup_task(cfg, args)

    for split in ["train", "test", "eval"]:
        build_fairseq_dataset(data_cfg, split, task.src_dict, task.tgt_dict, add_bos=True, add_eos=False)


# todo get dataset from wikidata