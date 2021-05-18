model=<multilingual model>
source_lang=input
target_lang=label
data_dir=/efs-storage/data-bin/fairseq_test/generate

fairseq-generate $data_dir \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $source_lang \
  --target-lang $target_lang
  --sacrebleu --remove-bpe 'sentencepiece'\
  --batch-size 32 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}.txt


cat {source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3- |$TOK_CMD > ${source_lang}_${target_lang}.hyp
cat {source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2- |$TOK_CMD > ${source_lang}_${target_lang}.ref
sacrebleu -tok 'none' -s 'none' ${source_lang}_${target_lang}.ref < ${source_lang}_${target_lang}.hyp

yuqint@amazon.com



# filter out [en_XX]
cat $inf_out | grep -P “^D” |sort -V |cut -f 3- | sacrebleu -l en-en $ref

cat $inf_out | sacrebleu -l en-en $ref