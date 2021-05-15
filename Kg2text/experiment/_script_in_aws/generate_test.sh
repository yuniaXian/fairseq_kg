fairseq-generate ~/efs-storage/data-bin/iwslt14.tokenized.de-en \
  --path ~/efs-storage/checkpoints/wmt14.en-fr.fconv-py/checkpoint_best.pt \
  --beam 5 \
  --remove-bpe
