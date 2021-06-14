create_dataset.py:
    + triples -> triples
    + text -> text
    + triples -> text (no mining in pretrain, just for finetune)
setting:
    + seperate (for triples), used for exacting triples in dbpedia and wikidata in the future
    + tagged (for triples)
    + bpe tokenized ( all)
implement:
    + initialize dataset
    + getitem
    + writ2file
entry:
    + triples -> triples
    + text -> text
    + triples -> text (no mining in pretrain, just for finetune)
# seperate/tagged/tokenized
    ## seperate: break each sample into a set of triples:
        denosing dataset: one triple per line
    ## with text label:
        lang pari dataset: joined triples per line
    ## tagged: tiples will be tagged [KG] [ENT] Sweet potato [PRED] main ingrredients [SUB] Binignit [TRIPLE]
    ## tokenized: words will be tokenized by bpe tokenizer

## dataset in fairseq:
```
data_path = paths[(epoch - 1) % len(paths)]
split_path = os.path.join(data_path, split)
languages: 
    if it is not given: list of all the dirs folders under data_path
    if it is given: list of all the dirs folders (langs) under data_path

data-bin:
    webnlg
        raw
        sentencepiece_bped
            en_XX
                KG:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt",
                TEXT:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt"
                NONE:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt
            zh_CN
            ...
```
data-bin:
    dataset:
        webnlg:
            train.json
    dataset_denoising
        webnlg:
            en_XX
                kg2kg:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt",
                text2text:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt"
                kg2text:
                    eval
                    train
                    test
                    "--train-subset",
                    "train.txt",
                    "--valid-subset",
                    "eval.txt
            zh_CN
        ...



