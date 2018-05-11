#!/bin/bash

FILE="./results.csv"
STRIPPED=`basename "$FILE" .csv`_stripped.csv
DST='.'

sed -e '/^$/d' <  "$FILE" > "$STRIPPED"
for type in `tail -n +2 < "$STRIPPED" | cut -d',' -f 3`
do
    type=`basename $type .TXT`
    echo "$type"
    cat <(head -n1 "$FILE") <(grep ROUGE-1 < "$STRIPPED" |  grep "$type" ) > "$DST"/ROUGE1_"$type".csv
    cat <(head -n1 "$FILE") <(grep ROUGE-2 < "$STRIPPED" |  grep "$type" ) > "$DST"/ROUGE2_"$type".csv
done
