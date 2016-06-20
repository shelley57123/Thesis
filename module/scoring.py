import math
import os
import numpy as np
from operator import itemgetter
from geopy.distance import great_circle

numK = 5
topK = []
hashmap = {}
hour = 8 #time for trip
T0 = 8 #default starting time

topK_cmp = []

User = 672/2 + 1

startClus = 3
startLm = 5
haveStartClus = True

clus_k = 30

popImp = 0.5
simImp = 0.2
ulmImp = 0.3

noSeq = False
noTime = False

seqtime = ''
if not noSeq:
    seqtime = seqtime+'Seq' 
if not noTime:
    seqtime = seqtime+'Time'

trans_hr = []
clus_time = []
clus_order = []

clus_hr_sort = []

"""estimate trans/clus time, order score"""
def estTransOrder(points, users, cluster_centers):

    transLen = 0.0
    transTimeLen = 0.0

    global trans_hr
    global clus_order
    transTimes = [] #number of times between each pair of clusters(no direction)
    for i in range(clus_k):
        trans_hr.append(np.zeros(clus_k))
        clus_order.append(np.ones(clus_k)*50)
    trans_hr = transTimes = np.array(trans_hr)
    clus_order = np.array(clus_order)

    global clus_time
    for i in range(clus_k):
        clus_time.append( np.ones(24)*50)
    clus_time = np.array(clus_time)

    allusers = np.unique(points[:,1])

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
                for i in range(l-1):

                    #find locations that have diff clus to next locaiton 
                    thisClus = (tsorted[same_date])[i][-2]
                    nextClus = (tsorted[same_date])[i+1][-2]

                    if (thisClus < clus_k):
                        clus_time[thisClus][(tsorted[same_date])[i][5]] += 1

                        if thisClus != nextClus and (nextClus < clus_k):
                            hr1 = (tsorted[same_date])[i][5] + (tsorted[same_date])[i][6]/60.0 #time of prior post
                            hr2 = (tsorted[same_date])[i+1][5] + (tsorted[same_date])[i+1][6]/60.0 #time of latter post
                            
                            thisLoc = ((tsorted[same_date])[i][10], (tsorted[same_date])[i][9])
                            nextLoc = ((tsorted[same_date])[i+1][10], (tsorted[same_date])[i+1][9])
                            transLen += great_circle(thisLoc, nextLoc).meters
                            transTimeLen += (hr2-hr1)

                            trans_hr[thisClus][nextClus] += (hr2-hr1)
                            trans_hr[nextClus][thisClus] += (hr2-hr1)

                            transTimes[thisClus][nextClus] += 1
                            transTimes[nextClus][thisClus] += 1

                            clus_order[thisClus][nextClus] += 1

    #speed unit: meters/hr
    avgSpeed = transLen / transTimeLen
    trans_hr = trans_hr / transTimes

    for i, row in enumerate(trans_hr):
        for j, timeLen in enumerate(row):
            if j > i:
                if np.isnan(timeLen) and i != j:
                    thisLoc = (cluster_centers[i][1], cluster_centers[i][0])
                    nextLoc = (cluster_centers[j][1], cluster_centers[j][0])
                    estTime = great_circle(thisLoc, nextLoc).meters / avgSpeed
                    
                    timeLen = estTime

                if round( (timeLen*100%100) / 50.0) == 0:
                    trans_hr[i][j] = trans_hr[j][i] = round(timeLen)
                elif round( (timeLen*100%100) / 50.0) == 1:
                    trans_hr[i][j] = trans_hr[j][i] = round(timeLen) + 0.5
                elif round( (timeLen*100%100) / 50.0) == 2:
                    trans_hr[i][j] = trans_hr[j][i] = round(timeLen) + 1

    print 'transition time:'
    print trans_hr

    clus_order = clus_order / np.max(clus_order)
    print 'condition probability:'
    print clus_order

    clus_time = clus_time / np.amax(clus_time, axis = 1)[:, None]
    print 'cluster time:'
    print clus_time

    # return [trans_hr, clus_time, clus_order]


"""Kullback-Leibler divergence D(P || Q) for discrete distributions
Parameters
----------
p, q : array-like, dtype=float, shape=n
Discrete probability distributions.
"""
def kl(p, q):

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    sum = 0

    for i in range(len(p)):
        if p[i] != 0:
            if q[i] > 1.0/1000000.0:
                sum += p[i] * np.log(p[i] / q[i])
            else:
                sum += p[i] * np.log(p[i] / (1.0/1000000.0) )
    return sum
    #return np.sum(np.where(p != 0, p * np.log(p / q), 0))


"""find landmarks to recommend of different clus-hr"""
def lmsOfClusHr(users, user_topic, doc_topic, points, t):

    global clus_hr_sort
    fileName = ' '.join(['data\\clus_hr_sort', str(popImp), str(simImp), str(ulmImp)])+'.txt'

    if os.path.isfile(fileName):
        f = open(fileName,'r')

        line = f.readline()
        clus_hr_num = int(line.split()[0])
        for i in range(clus_hr_num):
            clus, hr, score = f.readline().split()
            lms_num = int(f.readline().split()[0])
            lms = []
            for j in range(lms_num):
                lmId, lm_time, lm_score = f.readline().split()
                lms.append([int(float(lmId)), float(lm_time), float(lm_score)])
            clus_hr_sort.append([int(float(clus)), float(hr), float(score), lms])
        clus_hr_sort = sorted(clus_hr_sort, key=itemgetter(2), reverse = True ) 
        f.close()

    else:

        the_user = users[User]

        clus_hr_lms = [] #for diff hrs to a clus, recommend diff lms

        for lid in range(clus_k):
            the_clus = points[:,-2] == lid

            #lm of this clus
            lm_score = []

            if len(points[the_clus]) > 20:
                #estimate all scores of lms of this clus
                lm_score_sort = find_landmark_score(points, points[the_clus], the_user, users, user_topic, doc_topic, t)

                #find hrs of this clus
                dur_hr = []
                for user in points[the_clus, 1]:
                    if user in users:
                        user_points = points[the_clus,1] == user
                       
                        #for all points of the user, sort by time(index 2-7)
                        tsorted = np.array(sorted((points[the_clus])[user_points], key=itemgetter(2,3,4,5,6,7)))
                        
                        #dates of all posts of user
                        datesToPier = np.vstack({tuple(row) for row in tsorted[:,2:5]})#remove duplicate date
                        for date in datesToPier:
                            
                            #locations of the user visited in this date(boolean list)
                            same_date = tsorted[:,2:5] == date
                            sd = []
                            for d in same_date:
                                sd.append(d.all())
                            same_date = np.array(sd, dtype=bool)
                            
                            #add time difference of this day
                            hr2 = (tsorted[same_date])[-1][5] + (tsorted[same_date])[-1][6]/60.0 #last time of post today
                            hr1 = (tsorted[same_date])[0][5] + (tsorted[same_date])[0][6]/60.0 #first time of post today
                            dur_hr.append(round(hr2 - hr1))
                
                m = max(dur_hr)
                vec = np.zeros(m+1)
                for j in dur_hr:
                    vec[j] += 1
                vec = vec/sum(vec)

                std = np.std(dur_hr)
                
                for hr, p in enumerate(vec):
                    if p > 0.06 and hr > 0:
                
                        #KL divergence of specific hr
                        if std >= 1:
                            norm = np.random.normal(hr, std, 10000)
                        else:
                            norm = np.random.normal(hr, 1, 10000)

                        vec2 = np.zeros(m+1)
                        for j in norm:
                            if round(j,0) <= m and round(j,0) >= 0:
                                vec2[round(j,0)] += 1
                        vec2 = vec2/sum(vec2)
                        print 'vec2'
                        print vec2

                        #s = scipy.stats.entropy(pk=vec, qk=vec2, base=None)
                        s = kl(vec2, vec)
                        s2 = kl(vec, vec2)
                        kls = (s+s2)/2.0
                        print 'clus:'+str()
                        print 'hour:'+str(hr)
                        print 'KL:'+str(kls)

                        #choose lm
                        time_left = hr
                        lm_choose = []
                        for [lmId, lm_time, lm_score] in lm_score_sort:
                            if lm_time <= time_left:
                                lm_choose.append([lmId, lm_time, lm_score])
                                time_left -= lm_time
                                
                        #under this hr, sum of scores of chosen lms
                        if len(lm_choose) > 0:
                            lm_choose = np.array(lm_choose)
                            score = sum(lm_choose[:,2])
                            print 'score'+str(score)
                        
                            clus_hr_lms.append([lid, hr, score*kls, lm_choose])
                        print '\n'

        clus_hr_sort = sorted(clus_hr_lms, key=itemgetter(2), reverse = True )

        #write clus_hr file: list of [clus, hr, score, lms]
        clus_hr_file = open(fileName,'w')
        clus_hr_file.write(str(len(clus_hr_sort))+'\n')

        for clus, hr, score, lms in clus_hr_sort:
            clus_hr_file.write(str(clus)+' '+str(hr)+' '+str(score)+'\n')
            clus_hr_file.write(str(len(lms))+'\n')
            for lmId, lm_time, lm_score in lms:
                clus_hr_file.write(str(lmId)+' '+str(lm_time)+' '+str(lm_score)+'\n')

        clus_hr_file.close()

    return clus_hr_sort


def find_landmark_score(points, somePoints, the_user, users, user_topic, doc_topic, t):

    sim_users = []
    for user_idx, user in enumerate(users):
        if user != the_user:
            sim_users.append([user, np.dot(user_topic[User],user_topic[user_idx]) ])
    sim_sorted = np.array(sorted(sim_users, key=itemgetter(1),reverse=True ))

    lm_score = []
    #estimate all scores of lms of this clus
    landmarks = somePoints[:,-1]
    landmarks = np.unique(landmarks)
    for lmId in landmarks:
        the_lm = points[:,-1] == lmId
        landmark = points[the_lm]

        if len(landmark) > 20:
            #avg time of landmark

            dur_hr = []
            for user in points[the_lm, 1]:
                if user in users:
                    user_points = points[the_lm,1] == user
                   
                    #for all points of the user, sort by time(index 2-7)
                    tsorted = np.array(sorted((points[the_lm])[user_points], key=itemgetter(2,3,4,5,6,7)))
                    
                    #dates of all posts of user
                    datesToPier = np.vstack({tuple(row) for row in tsorted[:,2:5]})#remove duplicate date
                    for date in datesToPier:
                        
                        #locations of the user visited in this date(boolean list)
                        same_date = tsorted[:,2:5] == date
                        sd = []
                        for d in same_date:
                            sd.append(d.all())
                        same_date = np.array(sd, dtype=bool)
                        
                        #add time difference of this day
                        hr2 = (tsorted[same_date])[-1][5] + (tsorted[same_date])[-1][6]/60.0 #last time of post today
                        hr1 = (tsorted[same_date])[0][5] + (tsorted[same_date])[0][6]/60.0 #first time of post today
                        dur_hr.append(hr2 - hr1)
            
            if len(dur_hr)>0:
                lm_time = sum(dur_hr)/float(len(dur_hr))
                if round( (lm_time*100%100) / 50.0) == 0:
                    lm_time = round(lm_time)
                elif round( (lm_time*100%100) / 50.0) == 1:
                    lm_time = round(lm_time) + 0.5
                elif round( (lm_time*100%100) / 50.0) == 2:
                    lm_time = round(lm_time) + 1
                print lm_time
        
            # 1. popularity
            pop = len(landmark)/float(t) 
            print pop

            # 2. rate of similar people visiting lm
            n = 100
            c = 0
            landmark_users = points[the_lm, 1]
            
            topN_sim_users = sim_sorted[:n]
            for user in topN_sim_users:
                if user[0] in landmark_users:
                    c += 1
            sim = c/float(n)
            print sim

            # 3. user-landmark score
            ulm = np.dot(user_topic[User], doc_topic[lmId]) #similarity between user and landmark
            print ulm

            # 4. total
            #score = (pop * sim * ulm) ** (1/float(3))
            score = popImp*pop + simImp*sim + ulmImp*ulm
            print 'score: '+str(score)+'\n'

            lm_score.append([lmId, lm_time, score])
            
    lm_score_sort = np.array(sorted(lm_score, key=itemgetter(2,1) ))
    return lm_score_sort


"""for sorted clus_hr list, build a prefixDFS tree
update hashmap(K_plus:score,route,timeLen) simultaneously, recommend top K adequate trips at last
INPUT: U:list(clus_hr_sort[clus, hr, score, lms]), K:frozenset(clus, hr, score)"""
def prefixDFS(U,K):

    global topK
    global trans_hr
    global clus_time
    global clus_order
    haveK = False

    for i, clus_hr in enumerate(U):

        pre = U[0:i]
        Klist = np.array([list(x) for x in list(K)])

        thisClus, thisHr, thisScore = (clus_hr[0], clus_hr[1], clus_hr[2])

        K_plus = frozenset([(thisClus, thisHr, thisScore)]) | K
        #print K_plus

        if len(K) == 0: 
            if thisClus < clus_k and startClus != thisClus:

                if len(topK) >= numK:
                    cleanHashmap(topK[-1][0], clus_hr_sort)


                if haveStartClus:
                    timeLen = trans_hr[startClus][thisClus] + thisHr
                    #take the middle of visit as visiting hr (14:30 is credited to 14:00) rather than start time
                    ktime = T0 + trans_hr[startClus][thisClus] + math.floor(thisHr/2.0)
                    # print 'ktime '+str(ktime)
                    if noSeq:
                        cond = 1
                    else:
                        cond = clus_order[startClus][thisClus]
                    if noTime:
                        clusTime = 1
                    else:
                        clusTime = clus_time[thisClus][ktime]
                    kscore = thisScore * cond * clusTime
                else:
                    timeLen = thisHr
                    ktime = T0 + math.floor(thisHr/2.0)

                    if noTime:
                        clusTime = 1
                    else:
                        clusTime = clus_time[thisClus][ktime]

                    kscore = thisScore * clusTime

                if timeLen <= hour:
                    #keepRoute: pruning some routes that is impossible to become topK
                    #if (not haveK) or (haveK and keepRoute)
                    if not (haveK and not keepRoute(kscore, [ [thisClus, thisHr, thisScore] ], timeLen, topK[-1][0]) ):
                        #list of route(score, list of clus_hr_score with order, time of this order)
                        hashmap[K_plus] = [ [ kscore, [ [ thisClus, thisHr, thisScore ] ], timeLen ] ]

                    #if this route is long enough
                    if timeLen >= 0.9*hour:
                        topK.append([ kscore, [ [ thisClus, thisHr, thisScore ] ], timeLen ])
                        topK = sorted(topK, key=itemgetter(0),reverse=True )
                        if len(topK) == numK:
                            haveK = True
                            cleanHashmap(topK[-1][0], clus_hr_sort)
                        #only pick k routes with higher score 
                        topK = topK[:numK]

                    print str(i)+'_1st recursive'
                    prefixDFS(pre, K_plus)

        #prevent circumstances of clus has been chosen
        elif (thisClus not in Klist[:,0]):

            for k in K_plus:
                K_minus = K_plus - frozenset([k])
                
                if hashmap.has_key(K_minus):

                    for oldScore, oldRoute, oldTime in hashmap.get(K_minus):

                        preClus = (oldRoute)[-1][0]
                        kClus = k[0]
                        kHr = k[1]
                        kScore = k[2]

                        if preClus<clus_k and kClus<clus_k:

                            trans = trans_hr[ preClus ][ kClus ]
                            #total time of K_minus + time of visiting k + trans time from last clus to k clus
                            timeLen = oldTime + trans + kHr
                            
                            if (timeLen <= hour):
                                #take the middle of visit as visiting hr
                                ktime = T0 + oldTime + trans + math.floor(kHr/2.0)

                                if noSeq:
                                    cond = 1
                                else:
                                    cond = clus_order[preClus][kClus]
                                if noTime:
                                    clusTime = 1
                                else:
                                    clusTime = clus_time[kClus][ktime]
                                #original score + score of new clus (consider score of order and visiting hr)
                                newkScore = kScore * cond * clusTime
                                score = oldScore + newkScore

                                newRoute = list(oldRoute)
                                newRoute.append([ kClus, kHr, kScore ])

                                if hashmap.has_key(K_plus):
                                    if not (haveK and not keepRoute(score, newRoute, timeLen, topK[-1][0]) ):
                                        hashmap.get(K_plus).append([ score, newRoute, timeLen ])
                                else:
                                    if not (haveK and not keepRoute(score, newRoute, timeLen, topK[-1][0]) ):
                                        hashmap[K_plus] = [ [ score, newRoute, timeLen ]  ]

                                #consider transit time
                                if timeLen >= 0.9*hour:
                                    topK.append([ score, newRoute, timeLen ])

                                    if len(topK) == numK:
                                        haveK = True
                                        cleanHashmap(topK[-1][0], clus_hr_sort)
                                    #only pick k routes with higher score 
                                    topK = sorted(topK, key=itemgetter(0),reverse=True )[:numK]

            if hashmap.has_key(K_plus):
                prefixDFS(pre, K_plus)


def cleanHashmap(leastScore, clus_hr_sort):
    global hour
    delKeys = []
    for key, value in hashmap.iteritems():
        toDel = []
        for score, route, time in value:
            if not keepRoute(score, route, time, leastScore, clus_hr_sort):
                toDel.append([score, route, time])
        for score, route, time in toDel:
            value.remove([score, route, time])
        if len(value) == 0:
            delKeys.append(key)
    for key in delKeys:
        del hashmap[key]


def keepRoute(score, route, time, leastScore, clus_hr_sort):
    leftTime = hour-time
    maxClusNum = math.floor(leftTime)

    R = np.array(route)
    choose = []
    for clus, hr, s, lms in clus_hr_sort:
        if (clus not in R[:,0]) and (clus not in choose):
            choose.append(clus)
            score += s
        if len(choose) >= maxClusNum:
            break
    if score >= leastScore:
        return True
    else:
        return False



"""input:lm_score_sort list of[lmId, lm_time, score]"""
def cmp_method_generate_route(d, e, lm_score_sort, points):
    global topK_cmp
    k = 0
    Q = []
    # score = map_col(lm_score_sort, 0, 2, startLm)
    Q.append([ [startLm], 0, 0 ])
    while(k<numK):
        s = Q[0][0]
        # ds = route_time(s, points, lm_score_sort)
        ds = Q[0][1]
        scoreS = Q[0][2]
        Q.pop(0)
        if d-e <= ds and ds <= d+e :
            print s, ds, scoreS
            topK_cmp.append([s, ds, scoreS])
            k += 1
        if ds < d+e :
            for lm in lm_score_sort[:,0]:
                thisClus = map_col(points, -1, -2, lm)
                preClus = map_col(points, -1, -2, s[-1])
                if lm not in s and thisClus < clus_k and preClus < clus_k :
                    
                    newRoute = list(s)
                    newRoute.append(lm)
                    #find real preClus
                    if preClus == thisClus:
                        for i in range(len(s)-1):
                            if map_col(points, -1, -2, s[-2-i]) != thisClus :
                                preClus = map_col(points, -1, -2, s[-2-i])
                                break

                    if preClus != thisClus:
                        ktime = round( T0 + ds + trans_hr[preClus][thisClus] + map_col(lm_score_sort, 0, 1, lm)/2 )
                        newTime = ds + trans_hr[preClus][thisClus] + map_col(lm_score_sort, 0, 1, lm)
                        newScore = scoreS + map_col(lm_score_sort, 0, 2, lm)*\
                                    clus_order[preClus][thisClus]* clus_time[thisClus][ktime%24]
                    else:
                        ktime = round( T0 + ds + map_col(lm_score_sort, 0, 1, lm)/2 )
                        newTime = ds + map_col(lm_score_sort, 0, 1, lm)
                        newScore = scoreS + map_col(lm_score_sort, 0, 2, lm)* clus_time[thisClus][ktime%24]

                    if newTime <= d+e :
                        Q.append([newRoute, newTime, newScore])
                        Q = sorted(Q, key=itemgetter(2),reverse=True )
    return topK_cmp
                

# def route_time(route, points, lm_score_sort):
#     t = 0
#     for i, lm in enumerate(route):
#         #only transtime and order score
#         if i > 0:
#             thisClus = map_col(points, -1, -2, lm)
#             preClus = map_col(points, -1, -2, route[i-1])
#             t += trans_hr[preClus][thisClus] + map_col(lm_score_sort, 0, 1, lm)
#     return t


def map_col(matrix, search_col, return_col, val):
    for row in matrix:
        if row[search_col] == val:
            return row[return_col]
    # return matrix[ matrix[:,search_col].index(val), return_col ] #only for np array

