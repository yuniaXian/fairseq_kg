data_base=/home/ubuntu/efs-storage/data-bin
base=/home/ubuntu
efs=$base/efs-storage

for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json $efs/tokenizer/gpt2/bpe/encoder.json \
        --vocab-bpe $efs/tokenizer/gpt2/bpe/vocab.bpe \
        --inputs $data_base/wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs $data_base/wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done