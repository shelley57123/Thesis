import numpy as np
import ldaAdd
from sklearn.cluster import KMeans

topicNum = ldaAdd.topicNum
userClusNum = 10

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


def userClus(user_topic, topic_word, dic):
    km = KMeans(n_clusters=userClusNum, random_state=170)
    km.fit(user_topic)
    y_pred = km.labels_
    clus_topic = km.cluster_centers_

    for i in range(userClusNum):
        clus = y_pred == i
        user_topic = np.array(user_topic)
        users_topic = user_topic[clus]
        user_words = np.dot(users_topic, topic_word)
        user_words_mean = user_words.mean(0)

        n_top_words = 15
        user_top_words = np.array(dic)[np.argsort(user_words_mean)][:-(n_top_words+1):-1]
        print('Cluster {}: {}'.format(i, ' '.join(user_top_words)))

    return clus_topic