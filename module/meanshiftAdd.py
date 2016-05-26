import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from geopy.distance import great_circle

band = [0.008, 0.006]


"""1st layer"""
def ms1st(qua,loc):
    bandwidth = estimate_bandwidth(loc, quantile=qua, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(loc)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    return [labels, cluster_centers, n_clusters_, ms]


"""2nd layer"""
def ms2nd(BW_FILE,loc):
    global band
    if os.path.isfile(BW_FILE):
        f = open(BW_FILE,'r')
        for line in f.readlines():
            ls = line.split()
            if len(ls)>=1 and ls[0]!='':
	            bw = float(ls[0])
        
        labels2, cluster_centers2, n_clusters_2, ms2 = ms1st(bw,loc)

        return [labels2, cluster_centers2, n_clusters_2, ms2]

    else:
        avg = 500
        lev = -1
        gmap = True
        while avg>400 and lev < (len(band)-1):
            lev += 1
            # mymap2 = pygmaps.maps(37.78713, -122.42392, 13) 
            try:
                labels2, cluster_centers2, n_clusters_2, ms2 = ms1st(band[lev],loc)
                labels_copy = labels2 #2nd layer label

                print("number of estimated clusters : %d" % n_clusters_2)

                dist = []
                for i in range(n_clusters_2):
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
                    print avg
                else:
                    break
            except:
                gmap = False
        if gmap:
            
            f = open(BW_FILE,'w')
            f.write(str(band[lev]))
            f.close()

            return [labels2, cluster_centers2, n_clusters_2, ms2]