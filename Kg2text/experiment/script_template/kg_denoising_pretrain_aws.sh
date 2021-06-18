#!/bin/bash
CUDA=$1
EFS=/home/ubuntu/efs-storage
BASE=/home/ubuntu
WORKSPACE=$BASE
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text
TOKENIZER=${EFS}/tokenizer
#PRETRAIN=${EFS}/models/mbart50.ft.nn/model.pt
#langs=af_ZA,ar_AR,az_AZ,bn_IN,cs_CZ,de_DE,en_XX,es_XX,et_EE,fa_IR,fi_FI,fr_XX,gl_ES,gu_IN,he_IL,hi_IN,hr_HR,id_ID,it_IT,iu_CA,ja_JP,ja_XX,ka_GE,kk_KZ,km_KH,ko_KR,lt_LT,lv_LV,mk_MK,ml_IN,mn_MN,mr_IN,my_MM,ne_NP,nl_XX,pl_PL,ps_AF,pt_XX,ro_RO,ru_RU,si_LK,sl_SI,sv_SE,ta_IN,te_IN,th_TH,tr_TR,uk_UA,ur_PK,vi_VN,xh_ZA,zh_CN
#langs_25=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
# NAME=webnlg/data_mbart50_wtags


pretrain_setting=ft_mbart50/mask_15_mixed_span_35
#pretrain_setting=fs_mbart50/mask_15_mixed_span_35
#pretrain_setting=fs_mbart50/mask_15_word_word
DATADIR=${BASE}/dataset_denoising/kgtext_wikidata
PRETRAIN=${EFS}/models/mbart50.ft.nn/model_wtags0/model.pt
tensorboard_dir=$BASE/logs/tensorboard/denoising_kgtext_wikidata/$pretrain_setting
checkpoint_dir=$BASE/checkpoints/denoising_kgtext_wikidata/$pretrain_setting
restore_file=$checkpoint_dir/checkpoint2.pt

source /home/ubuntu/anaconda3/bin/activate pytorch_latest_p37
#python ${FAIRSEQ}/train.py ${DATADIR} \
CUDA_VISIBLE_DEVICES=${CUDA} python ${FAIRSEQ}/train.py ${DATADIR} \
    --encoder-normalize-before --decoder-normalize-before --arch mbart_large --task kg_multilingual_denoising  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --dataset-impl mmap  \
    --optimizer adam --adam-eps 1e-06 --adam-betas "(0.9, 0.98)"  \
    --lr-scheduler polynomial_decay --lr "1e-04" --stop-min-lr "-1"  \
    --warmup-updates 2500 --total-num-update 500000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 2560 --update-freq 2 --save-interval 1  --fp16 \
    --save-interval-updates 10000 --keep-interval-updates 20 --seed 222  \
    --validate-interval-updates 10000 \
    --log-format simple --log-interval 10 --save-dir $checkpoint_dir  \
    --layernorm-embedding --ddp-backend no_c10d --langs en_XX --no-whole-word-mask-langs False  \
    --sample-break-mode eos --whole_word_mask_mode word  \
    --mask 0.15 --mask-random 0.0 --insert 0.0  \
    --permute 0.0 --rotate 0.0 --poisson-lambda 3.5  \
    --permute-sentences 0.0 --mask-length span-poisson --replace-length "-1"  \
    --shorten-method none --bpe sentencepiece --sentencepiece-model /home/ubuntu/efs-storage/tokenizer/mbart50/bpe/sentence.bpe.model  \
    --train-subset train --valid-subset valid \
    --num-workers 8 --required-batch-size-multiple 8 \
    --tensorboard-logdir $tensorboard_dir \
    --restore-file $restore_file \
# --no-epoch-checkpoints
#     --reset-optimizer \
#  --finetune-from-model ${PRETRAIN} \
#  --memory-efficient-fp16 \
#----restore-file $PRETRAIN \
# --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler