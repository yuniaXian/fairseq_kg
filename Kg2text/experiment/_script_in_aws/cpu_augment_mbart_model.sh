#!/bin/sh
BASE=/home/ubuntu
python ${BASE}/fairseq/Kg2text/model_utils/augment_mbart_model.py \
   ${BASE}/efs-storage/models/mbart50.ft.nn \
   --tgt-dict ${BASE}/efs-storage/data-bin/dict.mbart50_wtags.txt \
   --save-to ${BASE}/efs-storage/models/mbart50.ft.nn/model_wtags
