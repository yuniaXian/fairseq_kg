cat train.json \
  | jq -cn --stream 'fromstream(1|truncate_stream(inputs))' \
  | split --line-bytes=1m --numeric-suffixes - train