
i = 0
pos = 0
neg = 0
f = open("predict_output_lg.csv", "r")
arr = {}
for line in f:
    if (i == 0):
        i += 1
        continue
    splited = line.rstrip().split(',')
    id = splited[0]
    v = 0
    if (float(splited[1]) > 0.62):
        v = 1
        pos += 1
    else:
        neg += 1
    arr[id] = v

of = open("predict_baoli.csv", "w")
of.write("Id,Prediction" + '\n')
j = 0
for (k, v) in arr.items():
    j += 1
    print(j)
    of.write(str(k) + ',' + str(v) + '\n')

print(pos)
print(neg)

