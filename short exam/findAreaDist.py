import numpy as np
import time
import pandas as pd
import os
import sys
from sklearn.cluster import MeanShift
from geopy.distance import great_circle

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
labels2, cluster_centers2, n_clusters_2, ms2 = msAdd.ms1st(0.015, loc)

dist = []
for i in range(n_clusters_2/2):
    my_members2 = labels2 == i
    if len(loc[my_members2, 0])>1:

        max_lat = 0
        min_lat = 90
        max_long = -180
        min_long = 0

        for j in range(0,len(loc[my_members2, 0])):
            # mymap2.addpoint(loc[my_members2, 1][j], loc[my_members2, 0][j], colors[i%7], title = str(i))

            # max - min
            if loc[my_members2, 1][j] > max_lat:
                max_lat = loc[my_members2, 1][j]
            if loc[my_members2, 1][j] < min_lat:
                min_lat = loc[my_members2, 1][j]
                
            if loc[my_members2, 0][j] > max_long:
                max_long = loc[my_members2, 0][j]    
            if loc[my_members2, 0][j] < min_long:
                min_long = loc[my_members2, 0][j]
                
        up = (max_lat, (max_long+min_long)/2)
        down = (min_lat, (max_long+min_long)/2)
        right = ((max_lat+min_lat)/2, max_long)  #lat(37),long(-122)
        left = ((max_lat+min_lat)/2, min_long)  #lat(37),long(-122)

        height = great_circle(up, down).meters
        length = great_circle(right, left).meters
        
        if height > length :
            max_l = height
        else :
            max_l = length
            
        print 'cluster'+str(i)+'_'+str(max_l)
        if max_l>0:
            dist.append(max_l)
if len(dist)>0:
    avg = sum(dist) / float(len(dist))
    print 'avg:'
    print avg




print '============End Time============'
print time.strftime('%Y-%m-%d %A %X',time.localtime(time.time())) 
print '================================'


# if __name__ == '__main__':
#     main()
