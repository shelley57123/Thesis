import numpy as np
import time
import pandas as pd
import os
import sys
from sklearn.cluster import MeanShift
import random

import string
import math
from sets import Set

# from operator import itemgetter

DIR = os.getcwd()
PAR_DIR = os.path.abspath('..')
sys.path.append(DIR)
sys.path.append(DIR+'\\module')
sys.path.append('..')

import ldaAdd
import meanshiftAdd as msAdd
import drawGmap
import scoring as sc
import plsaAdd
import userScore as us


FILE = './data/sf rm del excel2.csv'
DIC_FILE = './data/dic.txt'
BW_FILE = './data/bandwidth.txt'
CLUS_WORD_FILE = './data/ldac.txt'
CLUS_WORD_ZERO_FILE = './data/ldac_zero.txt'
USER_FILE = './data/user 30.txt'
ZW_FILE = './data/plsaZW.txt'
DZ_FILE = './data/plsaDZ.txt'

# def main():
print '===========Start Time==========='
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'

readFile = pd.read_csv(FILE, iterator=True, chunksize=1000, na_values = '')
points = pd.concat(readFile, ignore_index=True)

points = np.array(points, dtype=None)
loc = np.array(points[:,9:11],dtype=float)

if not os.path.isfile(DIC_FILE):
    dic = ldaAdd.dic(DIC_FILE, points)
else:
    dic = ldaAdd.readDic(DIC_FILE)

"""1st layer"""
labels, cluster_centers, n_clusters_, ms = msAdd.ms1st(0.015, loc)
print("number of estimated clusters in 1st layer: %d" % n_clusters_)
zeros = [0]*len(points)
#1st layer label
points = np.hstack((points, np.transpose([labels,zeros]) ))
#drawGmap.drawLayer(labels, cluster_centers, n_clusters_, loc, 1)


"""2nd layer"""
labels2, cluster_centers2, n_clusters_2, ms2 = msAdd.ms2nd(BW_FILE, loc)
print("number of estimated clusters in 2nd layer: %d" % n_clusters_2)

points[:,-1] = labels2
labelsNew = ms.predict(cluster_centers2) #landmark's new clus
for i in range(len(points)):
    #for each point, find it's landmark's new clus
    points[i,-2] = labelsNew[points[i,-1]]
#drawGmap.drawLayer(labels2, cluster_centers2, n_clusters_2, loc, 2)
clus_num = n_clusters_2


"""load matrix"""
if not os.path.isfile(CLUS_WORD_FILE):
    lda_m = ldaAdd.saveLda(clus_num, dic, points, CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)
    plsa_m = plsaAdd.savePlsa(clus_num, dic, points, CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)
else:
    lda_m = ldaAdd.readLda(CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)
    plsa_m = plsaAdd.readPlsa(clus_num, CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)

"""run lda/plsa"""
topic_word, doc_topic = ldaAdd.runLda(lda_m, dic)

if not (os.path.isfile(ZW_FILE) and os.path.isfile(DZ_FILE)):
    plsa_topic_word, plsa_doc_topic = plsaAdd.runPlsa(plsa_m, dic, CLUS_WORD_ZERO_FILE, ZW_FILE, DZ_FILE)
else:
    plsa_topic_word, plsa_doc_topic = plsaAdd.loadPlsa(ZW_FILE, DZ_FILE, clus_num, len(dic))

user_topic, plsa_user_topic, users, paths = us.userTopic_minus(USER_FILE, points, doc_topic, plsa_doc_topic)

"""trans/clus time, order score"""
sc.estTransOrder(points, users, cluster_centers)


rm_by_clus = points[:,-2] < sc.clus_k
somepoints = []
for x in points[rm_by_clus]:
    if x[1] in users:
        somepoints.append(x)
somepoints = np.array(somepoints)
print 'somepoints:'
print len(somepoints)


random.seed(120)
# randpaths = random.sample(paths, 400)
randusers = random.sample(users, 200)

"""estimate precision of one extraction"""
hit = 0
plsa_hit = 0
total = 0
for i, the_user in enumerate(randusers):
    user_paths = []
    sc.list_sim = []

    for user, diffClusIdx, diffLmIdx, path in paths:
        if user == the_user:
            user_paths.append([diffClusIdx, diffLmIdx, path])

    if len(user_paths) > 0:
        sc.clus_hr_sort = []
        clus_hr_sort = sc.lmsOfClusHr(users, user_topic, doc_topic, somepoints, user_topic[i], the_user)
        lm_score_sort = sc.find_all_lm(somepoints, users, plsa_user_topic, plsa_doc_topic, [], True, the_user)
        
        for diffClusIdx, diffLmIdx, path in user_paths:
            pastClus = np.unique(path[:diffClusIdx+1,-2])
            startClus = path[diffClusIdx,-2]

            T0 = path[diffClusIdx,5]+ round(path[diffClusIdx,6]/60.0)
            max_score = 0
            max_clus = -1
            for clus_hr in clus_hr_sort:
                thisClus, thisHr, thisScore = (clus_hr[0], clus_hr[1], clus_hr[2])

                if thisClus not in pastClus:

                    # timeLen = sc.trans_hr[startClus][thisClus] + thisHr
                    #take the middle of visit as visiting hr (14:30 is credited to 14:00) rather than start time
                    ktime = T0 + sc.trans_hr[startClus][thisClus] + math.floor(thisHr/2.0)
                    # print 'ktime '+str(ktime)
                    cond = sc.clus_order[startClus][thisClus]
                    clusTime = sc.clus_time[thisClus][ktime%24]

                    kscore = thisScore * cond * clusTime
                    if kscore > max_score:
                        max_clus = thisClus
                        max_score = kscore 

            if max_clus == path[-1,-2]:
                hit += 1
                
            print 'Hit, Total'
            print hit, total

            pastLm = np.unique(path[:diffLmIdx+1,-1])
            startLm = path[diffLmIdx,-1]
            T02 = path[diffLmIdx,5]+ round(path[diffLmIdx,6]/60.0)
            max_score = 0
            max_lm = -1
            for lm in lm_score_sort[:,0]:
                thisClus = sc.map_col(somepoints, -1, -2, lm)
                preClus = path[diffLmIdx,-2]

                thisLm_score = sc.map_col(lm_score_sort, 0, 2, lm)
                thisLm_hrs = sc.map_col(lm_score_sort, 0, 1, lm)
                thisClus_hrs = thisLm_hrs
                #find real preClus
                for j in range(diffClusIdx+1):
                    if path[-2-j,-2] != thisClus :
                        preClus = path[-2-j,-2]
                        break
                    else:
                        thisClus_hrs += sc.map_col(lm_score_sort, 0, 1, path[-2-j,-1])
                    thisClus_hrs = round(thisClus_hrs)

                if lm not in pastLm and thisClus < sc.clus_k and preClus < sc.clus_k :

                    m = sc.map_col(sc.clus_hrs_dist, 0, 1, thisClus)
                    std = sc.map_col(sc.clus_hrs_dist, 0, 2, thisClus)
                    vec = sc.map_col(sc.clus_hrs_dist, 0, 3, thisClus)

                    kls = sc.find_kl(vec, m, std, thisClus_hrs)

                    if preClus != thisClus:
                        ktime = round( T02 + sc.trans_hr[preClus][thisClus] + thisLm_hrs/2 )
                        newScore = thisLm_score* kls* \
                                    sc.clus_order[preClus][thisClus]* sc.clus_time[thisClus][ktime%24]
                    else:
                        ktime = round( T02 + thisLm_hrs/2 )
                        newScore = thisLm_score* kls* sc.clus_time[thisClus][ktime%24]

                    if newScore > max_score:
                        max_lm = lm
                        max_score = newScore

            if max_lm == path[-1,-1]:
                plsa_hit += 1
            total += 1

            print 'path, predict_clus, user, #userpaths, paths'
            print path, max_clus, i, len(user_paths), len(paths)
            print 'Hit, Total, Precision:'
            print hit, total, hit/float(total)
            print 'Plsa Hit, Precision:'
            print plsa_hit, plsa_hit/float(total)

print 'Precision of Total:'
print hit/float(total)


"""prefixDFS"""
sc.prefixDFS(clus_hr_sort, frozenset())
print 'TopK'
print sc.topK


print '============End Time============'
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'


# if __name__ == '__main__':
#     main()
