with open("./data/train_filtered.txt", 'r') as i:
    with open("./data/train_filtered_edgelist.ext",'w') as o:
        for line in i:
            s = line.rstrip().split("\t")
            if(len(s) > 1):
                src = s[0]
                sinks = s[1:]
                for e in sinks:
                    o.write(f"{src} {e}\n")