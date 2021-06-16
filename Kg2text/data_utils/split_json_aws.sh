EFS=/home/ubuntu/efs-storage
BASE=/home/ubuntu
WORKSPACE=$BASE
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text

dataset=webnlg
DATADIR=${EFS}/data-bin
load_data_dir=${DATADIR}/dataset
save_data_dir=${DATADIR}/dataset_kg2text/$dataset

SECONDS=0

cat $load_data_dir/$dataset/valid.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --lines=1000000 --numeric-suffixes - $save_data_dir/raw/valid

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


SECONDS=0
cat $load_data_dir/$dataset/test.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --lines=1000000 --numeric-suffixes - $save_data_dir/raw/test

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


SECONDS=0
cat $load_data_dir/$dataset/train.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --lines=1000000 --numeric-suffixes - $save_data_dir/raw/train

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