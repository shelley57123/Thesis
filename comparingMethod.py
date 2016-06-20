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
import scoring
import plsaAdd


FILE = './data/sf rm del excel2.csv'
DIC_FILE = './data/dic.txt'
BW_FILE = './data/bandwidth.txt'
CLUS_WORD_FILE = './data/ldac.txt'
CLUS_WORD_ZERO_FILE = './data/ldac_zero.txt'
MIDCLUS_FILE = './data/midclus.txt'
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


"""one layer"""
labels, cluster_centers, n_clusters_, ms = msAdd.ms2nd(BW_FILE, loc)
print("number of estimated clusters in 1st layer: %d" % n_clusters_)
midclus = open(MIDCLUS_FILE,'r')
midclus_list = [int(x) for x in midclus.readline().split()]
#layer label
points = np.hstack((points, np.transpose([midclus_list, labels]) ))
#drawGmap.drawLayer(labels, cluster_centers, n_clusters_, loc, 1)

clus_num = n_clusters_


"""run plsa"""
if not os.path.isfile(CLUS_WORD_FILE):
	plsa_m = plsaAdd.savePlsa(clus_num, dic, points, CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)
else:
	plsa_m = plsaAdd.readPlsa(clus_num, CLUS_WORD_FILE, CLUS_WORD_ZERO_FILE)

if not (os.path.isfile(ZW_FILE) and os.path.isfile(DZ_FILE)):
    topic_word, doc_topic = plsaAdd.runPlsa(plsa_m, dic, CLUS_WORD_ZERO_FILE, ZW_FILE, DZ_FILE)
else:
    topic_word, doc_topic = plsaAdd.loadPlsa(ZW_FILE, DZ_FILE, clus_num, len(dic))

print doc_topic.shape
user_topic, users, t = ldaAdd.userTopic(USER_FILE, points, doc_topic)

"""trans/clus time, order score"""
scoring.estTransOrder(points, users, cluster_centers)

U = len(users)/2 + 1
the_user = users[U]
lm_score_sort = scoring.find_landmark_score(points, points, the_user, users, user_topic, doc_topic, t)

topK_cmp = scoring.cmp_method_generate_route(8, 1, lm_score_sort, points) #K, d, e, lm_score_sort, points
print 'TopK'
print topK_cmp

drawGmap.drawTopK_cmp(topK_cmp, cluster_centers)

print '============End Time============'
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'


# if __name__ == '__main__':
#     main()
