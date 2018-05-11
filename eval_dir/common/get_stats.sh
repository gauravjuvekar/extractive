#./projects/format_opinosis.sh {enter_path}
mkdir -p statistics

java -jar rouge2-1.2.jar
mv results.csv statistics

cd statistics
./splitcsv.sh

python3 average.py
python3 max.py > ../statistics.txt

cd ..
#rm -rf ./statistics/*.csv
