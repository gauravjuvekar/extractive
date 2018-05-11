def average_column (csv):
    f = open(csv,"r")
    average = 0
    Sum = 0
    row_count = 0
    rec = []
    prec = []
    fsc = []
    f1 = f.readlines()
    f.close()
    f2 = f1[1:]
    for row in f2:
        r = row.split(',')
        rec.append(float(r[3]))
        prec.append(float(r[4]))
        fsc.append(float(r[5]))
        row_count += 1

    #print(row_count)

    sum = 0.0
    for i in range(len(rec)):
        sum += rec[i]
    rval = sum/row_count
    #print('REC: ' + str(rval))

    sum = 0.0
    for i in range(len(prec)):
        sum += prec[i]
    pval = sum/row_count
    #print('PREC: ' + str(pval))

    sum = 0.0
    for i in range(len(fsc)):
        sum += fsc[i]
    fval = sum/row_count
    #print('FSC: ' + str(fval))

    return (csv + ',' + str(rval) + ',' + str(pval) + ',' + str(fval))

import os
directory = '.'

n5 = open('./stats/COSINEORIGCASE-r1.txt',"w")
n6 = open('./stats/COSINEORIGCASE-r2.txt',"w")

n13 = open('./stats/EUCLIDEANORIGCASE-r1.txt',"w")
n14 = open('./stats/EUCLIDEANORIGCASE-r2.txt',"w")

print("COSINE ORIG CASE STATISTICS...")

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if("ROUGE1" in filename and "COSINE" in filename):
            #print(os.path.join(directory, filename))
            print(average_column(filename), file=n5)

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if("ROUGE2" in filename and "COSINE" in filename):
            #print(os.path.join(directory, filename))
            print(average_column(filename), file=n6)

print("EUCLIDEAN ORIG CASE STATISTICS...")

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if("ROUGE1" in filename and "EUCLIDEAN" in filename):
            #print(os.path.join(directory, filename))
            print(average_column(filename), file=n13)

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if("ROUGE2" in filename and "EUCLIDEAN" in filename):
            #print(os.path.join(directory, filename))
            print(average_column(filename), file=n14)

n5.close()
n6.close()
n13.close()
n14.close()
