


def dic(DIC_FILE, points):
    dic_out = open(DIC_FILE,'w')
    dic = []
    for i, val in enumerate(points):
        if isinstance(val[-1], str):
            tags = val[-1].split(' ')
            for tag in tags:
                if tag.find(':')==-1:            
                    try:
                        dic.index(tag)
                    except:
                        if len(tag)>0:
                            dic.append(tag)
                            dic_out.write(tag+'\n')
    #print dic
    dic_out.close()
    return dic


def readDic(DIC_FILE):
    dic_in = open(DIC_FILE,'r')
    dic = []
    for line in dic_in.readlines():
        ls = line.split()
        if len(ls)>0:
            dic.append(ls[0])
    return dic
