data_base=/home/ubuntu/efs-storage/data-bin
base=/home/ubuntu
efs=$base/efs-storage

fairseq-preprocess \
    --only-source \
    --srcdict $efs/tokenizer/gpt2/dict/dict.txt \
    --trainpref $data_base/wikitext-103-raw/wiki.train.bpe \
    --validpref $data_base/wikitext-103-raw/wiki.valid.bpe \
    --testpref $data_base/wikitext-103-raw/wiki.test.bpe \
    --destdir $data_base/wikitext-103 \
    --workers 60