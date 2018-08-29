of = open("data/twitter_test.txt", "a")
with open("data/test-public.txt", "r") as f:
    i = 0
    for line in f:
        if (i == 0):
            i += 1
            continue
        # split the line
        splited = line.rstrip().split('\t')
        v0 = splited[0]
        v1 = splited[1]
        v2 = splited[2]
        string = str(v0) + '\t' + str(v1) + '\t' + str(v2)
        of.write(string + '\n')
