log_base=/home/ubuntu/logs
nohup_output=$1
decoded_output=$2
tags=[en_XX]


cat $log_base/$nohup_output | grep -P "^D" |sort -V |cut -f 3- | cat > $log_base/$decoded_output.hyp
cat $log_base/$nohup_output | grep -P "^T" |sort -V |cut -f 2- |cat > $log_base/$decoded_output.ref
python "/home/ubuntu/fairseq/Kg2text/data_utils/del_tags.py" \
    --load_file $log_base/$decoded_output.hyp --save_file  $log_base/$decoded_output.hyp \
    --tags_to_del $tags

python "/home/ubuntu/fairseq/Kg2text/data_utils/del_tags.py" \
    --load_file $log_base/$decoded_output.ref --save_file  $log_base/$decoded_output.ref \
    --tags_to_del $tags

#./measure_scores.py example-inputs/devel-conc.txt example-inputs/baseline-output.txt