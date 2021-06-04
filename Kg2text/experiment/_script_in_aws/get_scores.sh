file=$1
cat $file.out | grep -P "^D" |sort -V |cut -f 3- > $file.hyp