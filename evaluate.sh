#!/bin/bash
set -x

OUT_DIR="$1"
rm -rf output_tmp
cp -r "$OUT_DIR" output_tmp

OP_DIR="output_tmp/opinosis"

for folder in $OP_DIR/*
do
    THIS_DIR=$(pwd)
    cd "$folder"
    rename -E 's/_//g' ./*.txt.data*
    rename -E 's/\.txt\.data(K[1-9]+)-(euclidean|cosine)/_'`basename $folder`'$2\.$1/' ./*.txt.data*
    cd "$THIS_DIR"
done


OP_DIR="output_tmp/cmplg"

for folder in $OP_DIR/*
do
    THIS_DIR=$(pwd)
    cd "$folder"
    rename -E 's/_//g' ./*.txt*
    rename -E 's/body\.txt(K[1-9]+)-(euclidean|cosine)/_'`basename $folder`'$2\.$1/' ./*.txt*
    cd "$THIS_DIR"
done


OPINOSIS_DIR=$(realpath ./data/opinosis/gold)
CMPLG_DIR=$(realpath ./data/cmplg-xml/gold)

OP_SIF=$(realpath ./output_tmp/opinosis/sif)
OP_S2V=$(realpath ./output_tmp/opinosis/s2v)
OP_SNP=$(realpath ./output_tmp/opinosis/sifnopcr)
CP_SIF=$(realpath ./output_tmp/cmplg/sif)
CP_S2V=$(realpath ./output_tmp/cmplg/s2v)
CP_SNP=$(realpath ./output_tmp/cmplg/sifnopcr)

cd "./eval_dir"
./init.sh OP_SIF "$OPINOSIS_DIR" "$OP_SIF"
./init.sh OP_S2V "$OPINOSIS_DIR" "$OP_S2V"
./init.sh OP_SNP "$OPINOSIS_DIR" "$OP_SNP"
./init.sh CP_SIF "$CMPLG_DIR" "$CP_SIF"
./init.sh CP_S2V "$CMPLG_DIR" "$CP_S2V"
./init.sh CP_SNP "$CMPLG_DIR" "$CP_SNP"
