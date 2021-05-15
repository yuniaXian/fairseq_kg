FAIRSEQ=/efs-storage/fairseq

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${FAIRSEQ}/train.py Kg2text/data-bin/webnlg/sentencepiece_bped_data \
--encoder-normalize-before --decoder-normalize-before --arch mbart_large --task  \
multilingual_denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --dataset-impl  \
raw --optimizer adam --adam-eps 1e-06 --adam-betas  \
"(0.9, 0.98)" --lr-scheduler polynomial_decay --lr 3e-05 --stop-min-lr  \
-1 --warmup-updates 2500 --max-update 40000 --total-num-update  \
40000 --dropout 0.3 --attention-dropout 0.1 --weight-decay  \
0.0 --max-tokens 1024 --update-freq 2 --save-interval  \
1 --save-interval-updates 8000 --keep-interval-updates 10 --no-epoch-checkpoints  \
--seed 222 --log-format simple --log-interval 2  \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --save-dir checkpoint/denoising_webnlg  \
--layernorm-embedding --ddp-backend no_c10d --langs en_XX --no-whole-word-mask-langs  \
False --tokens-per-sample 512 --sample-break-mode eos --mask  \
0.5 --mask-random 0.0 --insert 0.0 --permute  \
0.0 --rotate 0.5 --poisson-lambda 3.0 --permute-sentences  \
0.0 --mask-length word --replace-length -1 --shorten-method  \
none --bpe sentencepiece --sentencepiece-model Kg2text/tokenizer/mbart50/bpe/sentence.bpe.model --train-subset  \
train.input --valid-subset eval.input 