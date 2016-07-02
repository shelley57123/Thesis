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
import plsaAdd


FILE = './data/sf rm del excel2.csv'
DIC_FILE = './data/dic.txt'
BW_FILE = './data/bandwidth.txt'
CLUS_WORD_FILE = './data/ldac.txt'
CLUS_WORD_ZERO_FILE = './data/ldac_zero.txt'
USER_FILE = './data/user 30.txt'
ZW_FILE = './data/plsaZW.txt'
DZ_FILE = './data/plsaDZ.txt'
AVG_K_FILE = './data/avg_k.txt'
AVG_HR_FILE = './data/avg_hr.txt'
PARA_FILE = './data/para.txt'

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
else:
	lda_m = ldaAdd.readLda(CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)

"""run lda"""
topic_word, doc_topic = ldaAdd.runLda(lda_m, dic)
user_topic, users, users_pic_num = ldaAdd.userTopic(USER_FILE, points, doc_topic)

"""trans/clus time, order score"""
sc.estTransOrder(points, users, cluster_centers)

"""adjust weight beteen 3 parameter"""
for i in range(1,10):
	for j in range(1,11-i):
		para = open(PARA_FILE,'a')

		sc.popImp = i*0.1
		sc.simImp = j*0.1
		sc.ulmImp = 1.0 - sc.popImp - sc.simImp
		if sc.ulmImp < 0.05:
			sc.ulmImp = 0.0
		print sc.popImp, sc.simImp, sc.ulmImp

		clus_hr_sort = sc.lmsOfClusHr(users, user_topic, doc_topic, points, [])

		sc.topK = []
		sc.clus_hr_sort = []
		"""prefixDFS"""
		sc.prefixDFS(clus_hr_sort, frozenset())
		print 'TopK'
		print sc.topK

		para.write(str(sc.popImp)+' '+str(sc.simImp)+' '+str(sc.ulmImp)+' '+str(sc.topK_avg_score(sc.topK, 0))+'\n' )
		para.close()


print '============End Time============'
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'


# if __name__ == '__main__':
#     main()
