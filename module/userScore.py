import numpy as np
import ldaAdd
from sklearn.cluster import KMeans
from operator import itemgetter
import scoring as sc

topicNum = ldaAdd.topicNum
userClusNum = sc.userClusNum

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


def userTopic_minus(USER_FILE, points, doc_topic, plsa_doc_topic):

    uall = open(USER_FILE,'r')
    ucount = 0
    user_topic = []
    plsa_user_topic = []
    users = []
    paths = []
    t = 0
    for line in uall.readlines():
        ucount += 1
        ls = line.split()
        if len(ls)>=1 and ls[0]!='':
            users.append(ls[0])
            
    for user in users:
        user_points = points[:,1] == user
       
        #for all points of the user, sort by time(index 2-7)
        tsorted = np.array(sorted(points[user_points], key=itemgetter(2,3,4,5,6,7)))

        n = 0
        n2 = 0
        person_vec = np.zeros(topicNum, dtype=float)#topic amount
        plsa_person_vec = np.zeros(topicNum, dtype=float)#topic amount
        if len(tsorted)>0:

            #dates of all posts of user
            datesToClus = np.vstack({tuple(row) for row in tsorted[:,2:5]})#remove duplicate date
            for date in datesToClus:

                #locations of the user visited in this date(boolean list)
                same_date = tsorted[:,2:5] == date
                sd = []
                for d in same_date:
                    sd.append(d.all())
                same_date = np.array(sd, dtype=bool)
                
                #locations of posts in this date
                l = len(tsorted[same_date])
                path = tsorted[same_date]
                
                diffClusIdx = -1
                for i in range(l-1):
                    if path[-i-2,-2] != path[-1,-2]:
                        diffClusIdx = l-i-2
                
                diffLmIdx = -1
                for i in range(l-1):
                    if path[-i-2,-1] != path[-1,-1]:
                        diffLmIdx = l-i-2

                pastClus = np.unique(path[:diffClusIdx+1,-2])
                pastLm = np.unique(path[:diffLmIdx+1,-1])

                if l > 5 and diffClusIdx != -1 and path[diffClusIdx,-2] < sc.clus_k and \
                path[-1,-2] < sc.clus_k and path[-1,-2] not in pastClus and path[-1,-2] not in pastLm:
                    for val in path[:diffClusIdx+1]:
                        person_vec += doc_topic[ val[-1] ] 
                        n += 1
                    for val in path[:diffLmIdx+1]:
                        plsa_person_vec += plsa_doc_topic[ val[-1] ] 
                        n2 += 1
                    paths.append([user, diffClusIdx, diffLmIdx, path])

                else:
                    for val in path:
                        person_vec += doc_topic[ val[-1] ] 
                        plsa_person_vec += plsa_doc_topic[ val[-1] ] 
                        n += 1
                        n2 += 1

        print 'n:'
        print n
        print 'n2:'
        print n2

        person_vec =  person_vec / float(n)
        user_topic.append(person_vec)

        plsa_person_vec =  plsa_person_vec / float(n2)
        plsa_user_topic.append(plsa_person_vec)

    return [user_topic, plsa_user_topic, users, paths]