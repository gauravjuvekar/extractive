#!/bin/bash
set -x

TMP_FOLDER="$1"
GOLD_FOLDER=$(realpath "$2")
SYS_FOLDER=$(realpath "$3")

mkdir -p "$TMP_FOLDER"
cp -r common/* "$TMP_FOLDER"

THIS_FOLDER=$(pwd)
cd "$TMP_FOLDER/keys"
mkdir reference
mkdir system
cp $GOLD_FOLDER/* reference/
cp $SYS_FOLDER/* system/
cd "$THIS_FOLDER"

cd "$TMP_FOLDER"
./get_stats.sh
