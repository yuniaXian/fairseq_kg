#!/bin/bash
# specify directory: EFS, BASE, WORKSPACE
# choose raw dataset: webnlg/kgtext_wikidata/...
# set parameters:
# --option: kg2kg/kg2text/text2text 
# --seperate --text_only --tagged --tokenized --simple --lang --lang_tag
EFS=/home/xianjiay/efs-storage
BASE=/home/xianjiay
WORKSPACE=${EFS}/workspaces/hoverboard
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text
TOKENIZER=${EFS}/tokenizer
#PRETRAIN=${EFS}/models/mbart50.ft.nn/model.pt
#langs=af_ZA,ar_AR,az_AZ,bn_IN,cs_CZ,de_DE,en_XX,es_XX,et_EE,fa_IR,fi_FI,fr_XX,gl_ES,gu_IN,he_IL,hi_IN,hr_HR,id_ID,it_IT,iu_CA,ja_JP,ja_XX,ka_GE,kk_KZ,km_KH,ko_KR,lt_LT,lv_LV,mk_MK,ml_IN,mn_MN,mr_IN,my_MM,ne_NP,nl_XX,pl_PL,ps_AF,pt_XX,ro_RO,ru_RU,si_LK,sl_SI,sv_SE,ta_IN,te_IN,th_TH,tr_TR,uk_UA,ur_PK,vi_VN,xh_ZA,zh_CN
#langs_25=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
# NAME=webnlg/data_mbart50_wtags
DATADIR=${EFS}/data-bin
load_data_dir=${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg
save_data_dir=${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg/trial




fairseq-preprocess \
    --only-source \
    --srcdict ${TOKENIZER}/mbart50/dict/dict.mbart50_wtags.txt \
    --trainpref $load_data_dir/train01 \
    --destdir $save_data_dir \
    --all-gather-list-size 1310720 \
    --workers 100

mv ${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg/trial/train.idx ${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg/trial/train01.idx
mv ${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg/trial/train.bin ${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX/kg2kg/trial/train01.bin
#    --validpref $load_data_dir/valid \
#    --testpref $load_data_dir/test \