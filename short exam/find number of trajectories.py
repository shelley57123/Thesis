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

from operator import itemgetter

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

rm_by_clus = points[:,-2] < sc.clus_k
somepoints = []
for x in points[rm_by_clus]:
	if x[1] in users:
		somepoints.append(x)
somepoints = np.array(somepoints)
print 'somepoints:'
print len(somepoints)


n = 0
for user in users:
    user_points = points[:,1] == user
   
    #for all points of the user, sort by time(index 2-7)
    tsorted = np.array(sorted(points[user_points], key=itemgetter(2,3,4,5,6,7)))

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

            if l > 5:
                n += 1

print 'n:'
print n
