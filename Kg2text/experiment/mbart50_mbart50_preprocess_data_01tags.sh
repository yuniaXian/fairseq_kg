#!/usr/bin/env bash
DICT=/home/xianjiay/efs-storage/fairseq/data-bin/dict.mbart50.txt
#DICT=/home/xianjiay/efs-storage/fairseq/data-bin/dict.mbart50_wtags.txt
SRC=input
TGT=label
DATA=/home/xianjiay/efs-storage/fairseq/data-bin/webnlg
TRAIN=train
VALID=eval
TEST=test
DEST=/home/xianjiay/efs-storage/fairseq/data-bin/webnlg
NAME=wotags
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN} \
  --validpref ${DATA}/${VALID} \
  --testpref ${DATA}/${TEST} \
  --destdir ${DEST}/${NAME} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict= \
  --joined-dictionary \
  --workers 70