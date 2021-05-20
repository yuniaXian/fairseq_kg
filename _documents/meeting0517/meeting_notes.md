- [kgpt dataset:](#kgpt-dataset)
- [system env problem:](#system-env-problem)
- [dataset design:](#dataset-design)
- [experiment setting:](#experiment-setting)
- [decode setting:](#decode-setting)
- [script:](#script)
- [torch:](#torch)

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

# torch:
[filter_vals_in_tensor](https://intellipaat.com/community/19982/filter-a-tensor-on-the-basis-of-a-python-list-in-tensorflow)
```
import tensorflow as tf

with tf.Graph().as_default(), tf.Session() as sess:

    l = tf.constant([1, 2, 3], dtype=tf.int64)

    a = tf.constant([1, 2, 3, 4], dtype=tf.int64)

    m = tf.reduce_any(tf.equal(tf.expand_dims(a, 1), l), axis=1)

    b = tf.boolean_mask(a, m)

    print(sess.run(b))

```

```
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)




```
