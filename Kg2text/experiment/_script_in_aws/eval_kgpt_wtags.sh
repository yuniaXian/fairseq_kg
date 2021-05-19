base=/home/ubuntu
model=${base}/efs-storage/checkpoints/kgpt_train_webnlg_wtags_0/checkpoint_best.pt
data_dir=${base}/efs-storage/data-bin/webnlg/wotags
sentencepiece_model=${base}/efs-storage/tokenizer/mbart50/bpe/sentence.bpe.model
FAIRSEQ=${base}/fairseq/fairseq_cli

#fairseq-generate $data_dir \
CUDA_VISIBLE_DEVICES=${CUDA} python ${FAIRSEQ}/generate.py ${data_dir} \
  --path $model \
  --bpe 'sentencepiece' --sentencepiece-model ${sentencepiece_model} \
  --beam 5