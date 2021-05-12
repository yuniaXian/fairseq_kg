python /efs-storage/fairseq/Kg2text/model_utils/augment_mbart_model.py \
   /efs-storage/fairseq/Kg2text/model/mbart50.ft.nn \
   --tgt-dict "/efs-storage/fairseq/Kg2text/data-bin/dict.mbart50_wtags.txt" \
   --save-to "/efs-storage/fairseq/Kg2text/model/mbart50.ft.nn/model_wtags" \