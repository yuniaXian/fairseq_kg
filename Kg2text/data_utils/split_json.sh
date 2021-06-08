EFS=/home/xianjiay/efs-storage
BASE=/home/xianjiay
WORKSPACE=${EFS}/workspaces/hoverboard
FAIRSEQ=${WORKSPACE}/fairseq/fairseq_cli
KG2TEXT=${WORKSPACE}/fairseq/Kg2text

DATADIR=${EFS}/data-bin
load_data_dir=${DATADIR}/dataset
save_data_dir=${DATADIR}/dataset_denoising

cat $load_data_dir/kgtext_wikidata/train.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --lines=400000 --numeric-suffixes - $load_data_dir/kgtext_wikidata/train