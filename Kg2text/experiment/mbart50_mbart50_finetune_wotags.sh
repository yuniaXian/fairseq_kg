BASEDIR=~/efs-storage/fairseq
FAIRSEQ=${BASEDIR}/fairseq_cli
PRETRAIN=${BASEDIR}/Kg2text/model/mbart50.ft.nn/model.pt
#langs=af_ZA,ar_AR,az_AZ,bn_IN,cs_CZ,de_DE,en_XX,es_XX,et_EE,fa_IR,fi_FI,fr_XX,gl_ES,gu_IN,he_IL,hi_IN,hr_HR,id_ID,it_IT,iu_CA,ja_JP,ja_XX,ka_GE,kk_KZ,km_KH,ko_KR,lt_LT,lv_LV,mk_MK,ml_IN,mn_MN,mr_IN,my_MM,ne_NP,nl_XX,pl_PL,ps_AF,pt_XX,ro_RO,ru_RU,si_LK,sl_SI,sv_SE,ta_IN,te_IN,th_TH,tr_TR,uk_UA,ur_PK,vi_VN,xh_ZA,zh_CN
#langs_25=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=input
TGT=label
NAME=webnlg/wotags
# NAME=webnlg/data_mbart50_wtags
DATADIR=${BASEDIR}/Kg2text/data-bin/${NAME}
SAVEDIR=checkpoint/mbart50_mbart50_finetun_webnlg_wotags
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ${FAIRSEQ}/train.py
#python ${FAIRSEQ}/train.py ${DATADIR} \
fairseq-train ${DATADIR} \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --task translation --finetune-from-model ${PRETRAIN} --save-dir ${SAVEDIR} \
    --source-lang ${SRC} --target-lang ${TGT} \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2  \
    --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr "3e-05" --stop-min-lr '-1' \
    --warmup-updates 2500 --max-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 \
    --weight-decay 0.0 --max-tokens 1024 --update-freq 2 --save-interval 1 \
    --save-interval-updates 8000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --layernorm-embedding  --ddp-backend no_c10d
#--batch-size 32 \
#----restore-file $PRETRAIN \
# --langs ${langs}
# --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \