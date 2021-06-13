#!/bin/bash
# specify directory: EFS, BASE, WORKSPACE
# choose raw dataset: webnlg/kgtext_wikidata/...
# set parameters:
# --option: kg2kg/kg2text/text2text 
# --seperate --text_only --tagged --tokenized --simple --lang --lang_tag
source ~/anaconda3/bin/activate pytorch_latest_p37
#source ~/anaconda3/bin/activate py37

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
DATADIR=${EFS}/data-bin
load_data_dir=${BASE}/dataset_denoising/kgtext_wikidata/en_XX/tagged_tokenized
save_data_dir=${DATADIR}/dataset_denoising/kgtext_wikidata/en_XX

SECONDS=0


fairseq-preprocess \
    --only-source \
    --srcdict ${TOKENIZER}/mbart50/dict/dict.mbart50_wtags.txt \
    --validpref $load_data_dir/valid00 \
    --testpref $load_data_dir/test00 \
    --destdir $save_data_dir/temp \
    --workers 60


if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Current round takes $hours hour(s), $minutes minute(s) and $seconds second(s)" 
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Current round takes $minutes minute(s) and $seconds second(s)"
else
    echo "Current round takes $SECONDS seconds"
fi

i=0


<< comment
for file in $load_data_dir/valid*
do
        echo "Preprocess on" ${file}
        
        fairseq-preprocess \
                --only-source \
                --srcdict ${TOKENIZER}/mbart50/dict/dict.mbart50_wtags.txt \
                --validpref ${file} \
                --destdir $save_data_dir/temp \
                --workers 60
        
        echo "save to" $save_data_dir
        mv $save_data_dir/temp/valid.bin $save_data_dir/valid$i.bin
        mv $save_data_dir/temp/valid.idx $save_data_dir/valid$i.idx

        i=$((i+1))
done
comment

