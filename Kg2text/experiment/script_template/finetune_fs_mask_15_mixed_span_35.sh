CUDA=$1
EFS=/home/ubuntu/efs-storage
BASE=/home/ubuntu
WORKSPACE=$BASE
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text
TOKENIZER=${EFS}/tokenizer

#original PRETRAIN=${EFS}/models/mbart50.ft.nn/model.pt
#langs=af_ZA,ar_AR,az_AZ,bn_IN,cs_CZ,de_DE,en_XX,es_XX,et_EE,fa_IR,fi_FI,fr_XX,gl_ES,gu_IN,he_IL,hi_IN,hr_HR,id_ID,it_IT,iu_CA,ja_JP,ja_XX,ka_GE,kk_KZ,km_KH,ko_KR,lt_LT,lv_LV,mk_MK,ml_IN,mn_MN,mr_IN,my_MM,ne_NP,nl_XX,pl_PL,ps_AF,pt_XX,ro_RO,ru_RU,si_LK,sl_SI,sv_SE,ta_IN,te_IN,th_TH,tr_TR,uk_UA,ur_PK,vi_VN,xh_ZA,zh_CN
#langs_25=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

#pretrain_setting=ft_mbart50/mask_15_mixed_span_35
#pretrain_setting=fs_mbart50/mask_15_mixed_span_35
pretrain_setting=fs_mbart50/mask_15_word_word
checkpoint_file=checkpoint2.pt
dataset=webnlg
SRC=kg
TGT=text

DATADIR=${EFS}/data-bin/dataset_kg2text/$dataset/en_XX
PRETRAIN=${EFS}/checkpoints/denoising_kgtext_wikidata/$pretrain_setting/$checkpoint_file
tensorboard_dir=${BASE}/logs/tensorboard/kg2text_$dataset
checkpoint_dir=${BASE}/checkpoints/denoising_kgtext_wikidata/$pretrain_setting/ft_$dataset

source ~/anaconda3/bin/activate pytorch_latest_p37

#python ${FAIRSEQ}/train.py ${DATADIR} \
CUDA_VISIBLE_DEVICES=${CUDA} python ${FAIRSEQ}/train.py ${DATADIR} \
    --encoder-normalize-before --decoder-normalize-before --arch mbart_large --finetune-from-model ${PRETRAIN} \
    --dropout 0.3 --attention-dropout 0.1 --layernorm-embedding \
    --task translation --source-lang ${SRC} --target-lang ${TGT} \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2  \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --weight-decay 0.0 \
    --lr-scheduler inverse_sqrt --lr "3e-05" --stop-min-lr '-1' \
    --warmup-updates 2500 --max-update 40000 \
    --max-tokens 2560 --update-freq 1 \
	--fp16 --seed 222 \
    --validate-interval 10 \
    --save-interval 10 --save-dir ${checkpoint_dir} \
    --log-format simple --log-interval 10 --tensorboard-logdir $tensorboard_dir \
    --scoring bleu \
    --dataset-impl mmap --ddp-backend no_c10d \
    --num-workers 8 --required-batch-size-multiple 8 \
    
# --restore-file $PRETRAIN \
# --langs ${langs}
# --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
# --save-interval-updates 5000 --keep-interval-updates 5 \