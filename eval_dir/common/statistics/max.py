
l =  [
    './stats/SIFCOSINELOWERCASE-r1.txt',
    './stats/SIFCOSINELOWERCASE-r2.txt',
    './stats/S2VCOSINELOWERCASE-r1.txt',
    './stats/S2VCOSINELOWERCASE-r2.txt',
    './stats/SIFEUCLIDEANLOWERCASE-r1.txt',
    './stats/SIFEUCLIDEANLOWERCASE-r2.txt',
    './stats/S2VEUCLIDEANLOWERCASE-r1.txt',
    './stats/S2VEUCLIDEANLOWERCASE-r2.txt']
o = [
    './stats/SIFCOSINEORIGCASE-r1.txt',
    './stats/SIFCOSINEORIGCASE-r2.txt',
    './stats/SIFEUCLIDEANORIGCASE-r1.txt',
    './stats/SIFEUCLIDEANORIGCASE-r2.txt'
    ]


def max_stats(f1):
    f = open(f1, "r")
    average = 0
    Sum = 0
    row_count = 0
    rec = []
    prec = []
    fsc = []
    m = []
    f1 = f.readlines()
    f.close()
    for row in f1:
        r = row.split(',')
        m.append(r)
        rec.append(float(r[1]))
        prec.append(float(r[2]))
        fsc.append(float(r[3]))

    sum = 0.0
    mval = 0.0
    method = ''
    print("CALCULATING MAX RECALL")
    for i in range(len(rec)):
        if(rec[i] >= mval):
            mval = rec[i]
            method = m[i][0]
    print(method, mval)
    print()

    print("CALCULATING MAX PRECISION")
    sum = 0.0
    mval = 0.0
    method = ''
    for i in range(len(prec)):
        if(prec[i] >= mval):
            mval = prec[i]
            method = m[i][0]
    print(method, mval)
    print()

    print("CALCULATING MAX FSCORE")
    sum = 0.0
    rval = 0.0
    pval = 0.0
    mval = 0.0
    method = ''
    for i in range(len(fsc)):
        #print(fsc[i])
        if(fsc[i] >= mval):
            mval = fsc[i]
            rval = rec[i]
            pval = prec[i]
            method = m[i][0]
    print("CONCLUSION BASED ON FSCORE")
    print("MAX FSCORE == " + str(mval))
    print("*******************")
    print("FOR THIS METHOD...")
    print(method, rval, pval, mval)
    print("*******************")

print("RECALL PRECIS FSCORE")

def pf(f1):
    f = open(f1, "r")
    text = f.readlines()
    for l in text:
        print(l)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for fname in o:
    if(True):
        print("############ " + fname + " ############")
        print()
        max_stats(fname)
        print()
        print("#######################################")
    else:
        print(fname)
        pf(fname)
