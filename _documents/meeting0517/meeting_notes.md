- [kgpt dataset:](#kgpt-dataset)
- [system env problem:](#system-env-problem)
- [dataset design:](#dataset-design)
- [experiment setting:](#experiment-setting)
- [decode setting:](#decode-setting)
- [script:](#script)

# kgpt dataset:
How the samples loaded into model? \
[criterion](https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py#L79)

[fairseq_encoder](https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/fairseq_encoder.py#L43-L55)


# system env problem:

# dataset design:
concatenate dataset: triples-triples, triples-text

# experiment setting:
use max-tokens: > 4096, remove batch-size
lr decrease, 
use tensorboard to detect zigzag loss curves

# decode setting:
get text out and use metric to evaluate

# script:
yuqint@amazon.com
```
# filter out [en_XX]
cat $inf_out | grep -P “^D” |sort -V |cut -f 3- | sacrebleu -l en-en $ref
cat $inf_out | sacrebleu -l en-en $ref
```

```
cat {source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3- |$TOK_CMD > ${source_lang}_${target_lang}.hyp
cat {source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2- |$TOK_CMD > ${source_lang}_${target_lang}.ref
sacrebleu -tok 'none' -s 'none' ${source_lang}_${target_lang}.ref < ${source_lang}_${target_lang}.hyp

```


