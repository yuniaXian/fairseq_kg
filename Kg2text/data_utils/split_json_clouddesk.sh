EFS=/home/xianjiay/efs-storage
BASE=/home/xianjiay
WORKSPACE=$BASE/workspaces/hoverboard
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text

DATADIR=${EFS}/data-bin
load_data_dir=${DATADIR}/dataset
save_data_dir=${DATADIR}/dataset_denoising

SECONDS=0

cat $load_data_dir/kgtext_wikidata/valid.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --lines=400000 --numeric-suffixes - $load_data_dir/kgtext_wikidata/valid


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