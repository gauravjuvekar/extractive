#!/bin/bash

OUT_DIR="$1"
rm -f output_tmp
cp -r "$OUT_DIR" output_tmp

OP_DIR="output_tmp/opinosis"

for folder in "$OP_DIR/*"
do
    for file in "$folder/*"
    do
        echo $file
        rename -E 's/_//g' "$file"
        rename -E 's/\.txt\.data(K[1-9]+)-(euclidean|cosine)/_'"$folder"'$2\.$1/' "$file"
    done
done

OP_DIR="output_tmp/cmplg"

for folder in "$OP_DIR/*"
do
    for file in "$folder/*"
    do
        echo $file
        rename -E 's/_//g' "$file"
        rename -E 's/body\.txt(K[1-9]+)-(euclidean|cosine)/_'"$folder"'$2\.$1/' "$file"
    done
done

