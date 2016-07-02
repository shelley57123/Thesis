import numpy as np
import time
import pandas as pd
import os
import sys
from sklearn.cluster import MeanShift

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


FILE = './data/sf rm del excel2.csv'
DIC_FILE = './data/dic.txt'
BW_FILE = './data/bandwidth.txt'
LDA_FILE = './data/ldac.txt'
LDA_ZERO_FILE = './data/ldac_zero.txt'
MIDCLUS_FILE = './data/midclus.txt'
USER_FILE = './data/user 30.txt'


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
midclus = open(MIDCLUS_FILE,'w')
for i in range(len(points)):
    #for each point, find it's landmark's new clus
    points[i,-2] = labelsNew[points[i,-1]]
    midclus.write(str(points[i,-2])+' ')
midclus.close()
#drawGmap.drawLayer(labels2, cluster_centers2, n_clusters_2, loc, 2)
clus_num = n_clusters_2

"""run lda"""
if not os.path.isfile(LDA_FILE):
	lda_m = ldaAdd.saveLda(clus_num, dic, points, LDA_FILE, LDA_ZERO_FILE)
else:
	lda_m = ldaAdd.readLda(LDA_FILE, LDA_ZERO_FILE)

topic_word, doc_topic = ldaAdd.runLda(lda_m, dic)
user_topic, users, t = ldaAdd.userTopic(USER_FILE, points, doc_topic)

"""trans/clus time, order score"""
sc.estTransOrder(points, users, cluster_centers)

clus_hr_sort = sc.lmsOfClusHr(users, user_topic, doc_topic, points, [])

sc.prefixDFS(clus_hr_sort, frozenset())
print 'TopK'
print sc.topK

drawGmap.drawTopK(sc.topK, cluster_centers, cluster_centers2)

print '============End Time============'
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'


# if __name__ == '__main__':
#     main()
