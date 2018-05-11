#!/bin/bash

OUT_DIR="$1"
rm -rf output_tmp
cp -r "$OUT_DIR" output_tmp

OP_DIR="output_tmp/opinosis"

for folder in $OP_DIR/*
do
    THIS_DIR=$(pwd)
    cd "$folder"
    for file in ./*.txt.data*
    do
        echo $file
        rename -E 's/_//g' "$file"
    done
    for file in ./*.txt.data*
    do
        rename -E 's/\.txt\.data(K[1-9]+)-(euclidean|cosine)/_'`basename $folder`'$2\.$1/' "$file"
    done
    cd "$THIS_DIR"
done


OP_DIR="output_tmp/cmplg"

for folder in $OP_DIR/*
do
    THIS_DIR=$(pwd)
    cd "$folder"
    for file in ./*.txt.data*
    do
        echo $file
        rename -E 's/_//g' "$file"
    done
    for file in ./*.txt.data*
    do
        rename -E 's/body\.txt(K[1-9]+)-(euclidean|cosine)/_'`basename $folder`'$2\.$1/' "$file"
    done
    cd "$THIS_DIR"
done



