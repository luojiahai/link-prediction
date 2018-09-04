f = open("predict_feature_vectors.txt", "r")
of = open("test.csv", "w")
of.write("Id,Prediction" + '\n')
pos_n = 0
neg_n = 0
i = 0
for line in f:
    if (i == 0):
        i += 1
        continue
    splited = line.rstrip().split('\t')
    if ((float(splited[2]) > 0.5001) or (float(splited[3]) > 0.5001)):
        pos_n += 1
        of.write(str(i) + ',' + str(1) + '\n')
    else:
        neg_n += 1
        of.write(str(i) + ',' + str(0) + '\n')
    i += 1
print(pos_n)
print(neg_n)