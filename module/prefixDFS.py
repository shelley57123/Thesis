import math

def prefixDFS(U,K):

    global topK
    haveK = False

    for i, clus_hr in enumerate(U):

        # #防止K還沒有東西 或是 clus已經被選過的情況
        # if len(Klist) > 0 and clus_hr[0] not in Klist[:,0]:

        pre = U[0:i]

        Klist = np.array([list(x) for x in list(K)])
        #print Klist

        thisClus, thisHr, thisScore = (clus_hr[0], clus_hr[1], clus_hr[2])

        K_plus = frozenset([(thisClus, thisHr, thisScore)]) | K #clus, hr
        #print K_plus

        if len(K) == 0: 
            if thisClus <= clus_k and startClus != thisClus:

                if len(topK) >= numK:
                    cleanHashmap(topK[-1][0])


                if haveStartClus:
                    timeLen = trans_hr[startClus][thisClus] + thisHr
                    #取k的中間時間 14:30算成14:00的
                    # print 'thisClus '+str(thisClus)
                    # print 'thisHr '+str(thisHr)
                    ktime = T0 + trans_hr[startClus][thisClus] + math.floor(thisHr/2.0)
                    # print 'ktime '+str(ktime)
                    if noSeq:
                        cond = 1
                    else:
                        cond = condProb[startClus][thisClus]
                    if noTime:
                        clusTime = 1
                    else:
                        clusTime = clus_time[thisClus][ktime]
                    # kscore = thisScore * condProb[startClus][thisClus] * clus_time[thisClus][ktime]
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
                    if not (haveK and not keepRoute(kscore, [ [thisClus, thisHr, thisScore] ], timeLen, topK[-1][0]) ):
                        #list of route(X score, 順序的list clus_hr_score, 這組的時間, X 開始時間, X lms)
                        hashmap[K_plus] = [ [ kscore, [ [ thisClus, thisHr, thisScore ] ], timeLen ] ]

                    #如果現在這個clus_hr時間許可、可以選，而且剩下可用的時間已經足夠少
                    if timeLen >= 0.9*hour:
                        topK.append([ kscore, [ [ thisClus, thisHr, thisScore ] ], timeLen ])
                        topK = sorted(topK, key=itemgetter(0),reverse=True )
                        if len(topK) == numK:
                            haveK = True
                            cleanHashmap(topK[-1][0])
                        #只取分數大的K個
                        topK = topK[:numK]

                    print str(i)+'_1st recursive'
                    prefixDFS(pre, K_plus)

        #防止clus已經被選過的情況
        elif (thisClus not in Klist[:,0]):
            #print 'clus_hr not in Klist'

            for k in K_plus:
                K_minus = K_plus - frozenset([k])
                #print 'k in K_plus'
                #print K_plus
                #print k
                #print K_minus
                
                if hashmap.has_key(K_minus):
                    #print 'K_minus exist'
                    #print hashmap.get(K_minus)

                    for oldScore, oldRoute, oldTime in hashmap.get(K_minus):

                        preClus = (oldRoute)[-1][0]
                        kClus = k[0]
                        kHr = k[1]
                        kScore = k[2]

                        if preClus<clus_k and kClus<clus_k:

                            trans = trans_hr[ preClus ][ kClus ]
                            # if math.isnan(trans):
                            #     trans = 0
                            #     print 'nan:'+str(preClus)+' '+str(kClus)
                            #K_minus 的總時間 + k的小時數 + 最後一點的clus到k的交通時間
                            timeLen = oldTime + trans + kHr
                            
                            if (timeLen <= hour):
                                #k的中間時間
                                ktime = T0 + oldTime + trans + math.floor(kHr/2.0)

                                if noSeq:
                                    cond = 1
                                else:
                                    cond = condProb[preClus][kClus]
                                if noTime:
                                    clusTime = 1
                                else:
                                    clusTime = clus_time[kClus][ktime]
                                #原本分數 加上新的clus的分數 (並考慮順序性,時間合適性的分數)
                                newkScore = kScore * cond * clusTime
                                score = oldScore + newkScore

                                newRoute = list(oldRoute)
                                newRoute.append([ kClus, kHr, kScore ])

                                if hashmap.has_key(K_plus):
                                    if not (haveK and not keepRoute(score, newRoute, timeLen, topK[-1][0]) ):
                                        hashmap.get(K_plus).append([ score, newRoute, timeLen ])
                                else:
                                    if not (haveK and not keepRoute(score, newRoute, timeLen, topK[-1][0]) ):
                                        hashmap[K_plus] = [ [ score, newRoute, timeLen ]  ]#score, lms

                                #要考慮交通時間
                                if timeLen >= 0.9*hour:
                                    #print '5 loop'
                                    topK.append([ score, newRoute, timeLen ])

                                    if len(topK) == numK:
                                        haveK = True
                                        cleanHashmap(topK[-1][0])
                                    #只取分數大的K個
                                    topK = sorted(topK, key=itemgetter(0),reverse=True )[:numK]

            if hashmap.has_key(K_plus):
                #print "n recursive"
                prefixDFS(pre, K_plus)


def cleanHashmap(leastScore):
    global hour
    delKeys = []
    for key, value in hashmap.iteritems():
        toDel = []
        for score, route, time in value:
            if not keepRoute(score, route, time, leastScore):
                toDel.append([score, route, time])
        for score, route, time in toDel:
            value.remove([score, route, time])
        if len(value) == 0:
            delKeys.append(key)
    for key in delKeys:
        del hashmap[key]


def keepRoute(score, route, time, leastScore):
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
