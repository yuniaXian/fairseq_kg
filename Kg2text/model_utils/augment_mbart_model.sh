python /home/xianjiay/efs-storage/fairseq/Kg2text/model_utils/augment_mbart_model.py \
   /home/xianjiay/efs-storage/fairseq/Kg2text/model/mbart50.ft.nn \
   --tgt-dict "/home/xianjiay/efs-storage/fairseq/Kg2text/data-bin/dict.mbart50_wtags.txt" \
   --save-to "/home/xianjiay/efs-storage/fairseq/Kg2text/model/mbart50.ft.nn/model_wtags"