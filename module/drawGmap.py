import pygmaps 
import webbrowser 
import scoring as sc
import matplotlib.pyplot as plt
from geopy.distance import great_circle

colors = ['#FF0000','#0000FF','#00FF00','#00FFFF','#FFCD00','#FF00FF','#808080','#FFFFFF','#000000']

def drawLayer(labels, cluster_centers, n_clusters_, loc, layerNum):
	mymap = pygmaps.maps(37.78713, -122.42392, 13) 
	for k in range(n_clusters_):
	    my_members = labels == k
	    for j in range(0,len(loc[my_members, 0])):
	        mymap.addpoint(loc[my_members, 1][j], loc[my_members, 0][j], colors[k%7], title = str(k))
	    # mymap.addpoint(cluster_centers[k][1], cluster_centers[k][0], colors[8], title = str(k))
	pageName = 'sf layer'+str(layerNum)+'.html'
	mymap.draw(pageName)
	url = pageName
	webbrowser.open_new_tab(url)


def drawTopK(topKPath, cluster_centers, cluster_centers2):
    global colors
    mymap = pygmaps.maps(37.78713, -122.42392, 13)
    i = 0
    colorNum = 6
    for score, route, timeLen in topKPath:
        path = []
        if sc.haveStartClus:
            mymap.addpoint(cluster_centers[sc.startClus][1], cluster_centers[sc.startClus][0], colors[8], title = str(sc.startClus)+'_clus ')
            path.append((cluster_centers[sc.startClus][1], cluster_centers[sc.startClus][0]))

        for clus, hr, clusScore in route:
            lms = findClusLms(clus, hr)
            # mymap.addpoint(cluster_centers[clus][1], cluster_centers[clus][0], colors[i%colorNum], title = str(clus)+'_clus '+str(hr)+'hr')
            # path.append((cluster_centers[clus][1], cluster_centers[clus][0]))
            for lmId, lm_time, lm_score in lms:
                lon, lat = findLmLoc(lmId, cluster_centers2)
                if lmId != 186 and lmId!= 310:
                    if len(path) == 0:
                        mymap.addpoint(lat, lon, colors[8], title = str(lmId)+'_lm '+str(lm_time)+'hr')
                    else:
                        mymap.addpoint(lat, lon, colors[i%colorNum], title = str(lmId)+'_lm '+str(lm_time)+'hr')
                    path.append((lat, lon)) #draw lm paths
        mymap.addpoint(cluster_centers[clus][1], cluster_centers[clus][0], colors[7], title = str(lmId)+'_lm '+str(lm_time)+'hr')
        mymap.addpath(path, colors[i%colorNum])
        i += 1

    if sc.haveStartClus:
        drawName = '_'.join(['topK', str(sc.popImp), str(sc.simImp), str(sc.ulmImp), 'StartClus'+str(sc.startClus), sc.seqtime]) + '.html'
    else:
        drawName = '_'.join(['topK', str(sc.popImp), str(sc.simImp), str(sc.ulmImp), 'NoStart', sc.seqtime]) + '.html'
    mymap.draw(drawName)
    url = drawName
    webbrowser.open_new_tab(url)

#output: (long, lat) of landmark's center
def findLmLoc(lm, cluster_centers2):
    return cluster_centers2[lm]

#output: list of landmarks(lmId, lm_time, lm_score)
def findClusLms(inClus, inHr):
    for clus, hr, score, lms in sc.clus_hr_sort:
        if clus == inClus and hr == inHr:
            return lms


def drawTopK_cmp(topKPath, cluster_centers):
    global colors
    mymap = pygmaps.maps(37.78713, -122.42392, 13)
    i = 0
    colorNum = 6
    for route, timeLen, score in topKPath:
        path = []

        # mymap.addpoint(cluster_centers[sc.startLm][1], cluster_centers[sc.startLm][0], colors[8], title = str(sc.startLm)+'_clus ')
        # path.append((cluster_centers[sc.startLm][1], cluster_centers[sc.startLm][0]))

        for lmId in route:
            lon, lat = findLmLoc(lmId, cluster_centers)
            if len(path) == 0:
                mymap.addpoint(lat, lon, colors[8], title = str(lmId)+'_lm '+str(timeLen)+'hr s:'+str(score))
            else:
                mymap.addpoint(lat, lon, colors[i%colorNum], title = str(lmId)+'_lm ')
            path.append((lat, lon))

        mymap.addpoint(lat, lon, colors[7], title = str(lmId)+'_lm ')
        mymap.addpath(path, colors[i%colorNum])
        i += 1

    drawName = '_'.join(['topK_cmp', str(sc.popImp), str(sc.simImp), str(sc.ulmImp), 'StartLm'+str(sc.startLm) ]) + '.html'
    
    mymap.draw(drawName)
    url = drawName
    webbrowser.open_new_tab(url)


def plotHist(aList, i):

    plt.figure(i)
    plt.clf()
    plt.hist(aList)
    # plt.xlabel("Hours")
    # plt.ylabel("Frequency")
    plt.show()