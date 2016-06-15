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
LDA_FILE = './data/ldac.txt'
LDA_ZERO_FILE = './data/ldac_zero.txt'
USER_FILE = './data/user 30.txt'


def main():
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
    labels, cluster_centers, n_clusters_, ms = msAdd.ms1st(0.006, loc)
    print("number of estimated clusters in 1st layer: %d" % n_clusters_)
    zeros = [0]*len(points)
    #layer label
    points = np.hstack((points, np.transpose([labels,zeros]) ))
    #drawGmap.drawLayer(labels, cluster_centers, n_clusters_, loc, 1)
    clus_num = n_clusters_
    

    """run lda"""
    if not os.path.isfile(LDA_FILE):
    	plsa_m = plsaAdd.savePlsa(clus_num, dic, points, LDA_FILE, LDA_ZERO_FILE)
    else:
    	plsa_m = plsaAdd.readPlsa(LDA_FILE, LDA_ZERO_FILE)

    topic_word, doc_topic = plsaAdd.runPlsa(plsa_m, dic)
    user_topic, users, t = ldaAdd.userTopic(USER_FILE, points, doc_topic)

    # """trans/clus time, order score"""
    # scoring.estTransOrder(points, users, cluster_centers)

    # clus_hr_sort = scoring.lmsOfClusHr(users, user_topic, doc_topic, points, t)

    # scoring.prefixDFS(clus_hr_sort[:35], frozenset())
    # print 'TopK'
    # print scoring.topK

    # drawGmap.drawTopK(scoring.topK[:6], cluster_centers, cluster_centers2)

    print '============End Time============'
    print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
    print '================================'


if __name__ == '__main__':
    main()
