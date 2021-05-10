python ~/efs-storage/fairseq/experiment/model_utils/augment_mbart_model.py \
   ~/efs-storage/fairseq/checkpoint/mbart50.ft.nn \
   --tgt-dict "/home/xianjiay/efs-storage/fairseq/data-bin/dict.mbart50_wtags.txt" \
   --save-to "/home/xianjiay/efs-storage/fairseq/checkpoint/mbart50_mbart50_finetune_webnlg_wtags" \