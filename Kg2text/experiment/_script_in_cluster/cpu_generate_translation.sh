model=/efs-storage/checkpoint/mbart50_mbart50_finetun_webnlg_wtags/checkpoint_best.pt
data_dir=/efs-storage/data-bin/webnlg/wtags
sentencepiece_model=/efs-storage/tokenizer/mbart50/bpe/sentence.bpe.model

fairseq-generate $data_dir \
  --path $model \
  --bpe 'sentencepiece' --sentencepiece-model ${sentencepiece_model} \
  --beam 5