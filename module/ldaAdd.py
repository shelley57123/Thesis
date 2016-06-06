import numpy as np
import lda
import lda.utils

topicNum = 20

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
    dic_in.close()
    return dic


def saveLda(clus_num, dic, points, LDA_FILE, LDA_ZERO_FILE):

    lda_m = [[0]*len(dic)]*(clus_num)
    lda_m = np.array(lda_m)

    for i in range(len(points)):
        if isinstance(points[i][-3], str):
            tags = points[i][-3].split(' ')
            for tag in tags:
                if tag.find(':')==-1 and tag != '':
                    lda_m[ int(points[i][-1]) ][dic.index(tag)] += 1
    
    #exclude top 8 frequent words
    freq_words = np.array(dic)[np.argsort(np.sum(lda_m, axis=0))][:-(7+1):-1]
    print ' '.join(freq_words)
    zero = np.zeros(lda_m.shape[0]) #row of zeros of len(clus)
    tlda = np.transpose(lda_m) #clus*word to word*clus
    #set frequency of frequent words to zero
    for i in np.argsort(np.sum(lda_m, axis=0))[:-(7+1):-1]:
        tlda[i] = zero
    print np.array(tlda)[np.argsort(np.sum(lda_m, axis=0))][:-(7+1):-1]
    lda_m = np.transpose(tlda)#word*clus to clus*word

    clus_shift = zero
    ldac = open(LDA_FILE,'w')
    for i in range(lda_m.shape[0]):
        s = ''
        n = 0
        shift = 0
        for j in range(lda_m.shape[1]):
            if lda_m[i][j]!=0:
                n += 1
                s = s + ' '+str(j)+':'+str(lda_m[i][j])
        if n != 0:
            ldac.write(str(n)+s+'\n')
            #clus_shift[i] = shift
        else:
            shift += 1
            clus_shift[i] = -1
    ldac.close()

    lda_zero = open(LDA_ZERO_FILE,'w')
    for i in range(len(clus_shift)):
        if clus_shift[i] == -1:
            lda_zero.write(str(i)+'\n')
    lda_zero.close()

    return lda_m


def readLda(LDA_FILE, LDA_ZERO_FILE):
    lda_m = lda.utils.ldac2dtm(open(LDA_FILE), offset=0)

    zero = np.zeros(lda_m.shape[1]) #row of zeros of len(words)
    zero_idx = []
    lda_zero = open(LDA_ZERO_FILE,'r')
    for line in lda_zero.readlines():
        ls = line.split()
        if len(ls)>0:
            zero_idx.append(int(ls[0]))
    lda_zero.close()

    for i, idx in enumerate(zero_idx):
        lda_m = np.insert(lda_m, idx, zero, 0)

    return lda_m


def runLda(lda_m, dic):

    model = lda.LDA(n_topics=topicNum, n_iter=1500, random_state=1)
    model.fit(lda_m)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(dic)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_

    return [topic_word, doc_topic]


def userTopic(USER_FILE, points, doc_topic):

    uall = open(USER_FILE,'r')
    ucount = 0
    user_topic = []
    users = []
    t = 0
    for line in uall.readlines():
        ucount += 1
        ls = line.split()
        if len(ls)>=1 and ls[0]!='':
            users.append(ls[0])
            
            ufile = ls[0]+'.txt'
            u = open('data\\users\\'+ufile,'r')
            photoid = []
            for line2 in u.readlines():
                ls2 = line2.split(',')
                if len(ls2)>=1 and ls2[0]!='':
                    photoid.append(ls2[0])
                    
            n = len(photoid)
            t += n
            person_vec = np.zeros(topicNum, dtype=float)#topic amount
            for i, val in enumerate(points):
                #print val[0]
                #print photo
                if str(val[0]) in photoid:
                    person_vec += doc_topic[ points[i][-1] ] / float(n)

            #print ucount
            user_topic.append(person_vec)

    return [user_topic, users, t]