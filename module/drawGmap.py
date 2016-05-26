import pygmaps 
import webbrowser 

colors = ['#FF0000','#0000FF','#00FF00','#00FFFF','FFFF00','#FF00FF','#808080','#FFFFFF','#000000']

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


def drawTopK(topKPath):
    global colors
    mymap = pygmaps.maps(37.78713, -122.42392, 13)
    i = 0
    colorNum = 6
    for score, route, timeLen in topKPath:
        path = []
        if haveStartClus:
            mymap.addpoint(cluster_centers[startClus][1], cluster_centers[startClus][0], colors[8], title = str(startClus)+'_clus ')
            path.append((cluster_centers[startClus][1], cluster_centers[startClus][0]))

        for clus, hr, clusScore in route:
            lms = findClusLms(clus, hr)
            mymap.addpoint(cluster_centers[clus][1], cluster_centers[clus][0], colors[i%colorNum], title = str(clus)+'_clus '+str(hr)+'hr')
            for lmId, lm_time, lm_score in lms:
                lon, lat = findLmLoc(lmId)
                if len(path) == 0:
                    mymap.addpoint(lat, lon, colors[8], title = str(lmId)+'_lm '+str(lm_time)+'hr')
                else:
                    mymap.addpoint(lat, lon, colors[i%colorNum], title = str(lmId)+'_lm '+str(lm_time)+'hr')
                path.append((lat, lon))
        mymap.addpoint(lat, lon, colors[7], title = str(lmId)+'_lm '+str(lm_time)+'hr')
        mymap.addpath(path, colors[i%colorNum])
        i += 1

    if haveStartClus:
        drawName = '_'.join(['topK', str(popImp), str(simImp), str(ulmImp), 'StartClus'+str(startClus), seqtime]) + '.html'
    else:
        drawName = '_'.join(['topK', str(popImp), str(simImp), str(ulmImp), 'NoStart', seqtime]) + '.html'
    mymap.draw(drawName)
    url = drawName
    webbrowser.open_new_tab(url)