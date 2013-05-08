import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import sklearn
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import igraph
import math

import directories
from visualization import *

ImagesAndBBoxes = directories.loadImagesAndBBoxes()
directories.ensure_dir(directories.output)

def calcMaskUsingMyGrabCut(img, bbox, filename):
    trimap = np.ones(img.shape[:2])
    trimap *= -1
    
    for x in range(bbox[0], bbox[2]):
        for y in range(bbox[1], bbox[3]):
            trimap[y, x] = 0
    
    mask = np.copy(trimap)
    mask += 1
    #visualize(mask)
    
    for iteration in range(10):
        fgObs = img[mask == 1]
        bgObs = img[mask == 0]
        allObs = np.asarray(list(fgObs) + list(bgObs))
        """
        #Plot the point clouds
        #plt.scatter(x=allObs[::100, 2], y=allObs[::100, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        step = 10000
        
        colorList = [0] * len(fgObs) + [1] * len(bgObs)
        colorArr = np.zeros((len(allObs), 3))
        
        colorArr[:, 0] = 255 *  np.asarray(colorList)
        colorArr[:, 1] = 255 * (1 - np.asarray(colorList))
        colorArr[:, 2] = 255
        
        ax.scatter(allObs[::step, 0], allObs[::step, 1], allObs[::step, 2], c=colorArr)
        
        plt.show()   
        exit()
        """
        
        numComponents = 10
        
        print("Making GMM...")
        
        if len(fgObs) >= numComponents:
            #Make FG Components
            fgGMM = sklearn.mixture.DPGMM(n_components=numComponents, alpha=.001, covariance_type='full', random_state=None, thresh=0.01, min_covar=None, n_iter=10, params='wmc', init_params='wmc')
            fgGMM.fit(fgObs)
            
            fgComponents = fgGMM.predict(np.reshape(img, (-1, 3)))
            fgComponents = np.reshape(fgComponents, mask.shape)
            
        if len(bgObs) >= numComponents:
            #Make BG Components
            bgGMM = sklearn.mixture.DPGMM(n_components=numComponents, alpha=.001, covariance_type='full', random_state=None, thresh=0.01, min_covar=0.001, n_iter=10, params='wmc', init_params='wmc')    
            bgGMM.fit(bgObs)
            
            bgComponents = bgGMM.predict(np.reshape(img, (-1, 3)))
            bgComponents = np.reshape(bgComponents, mask.shape)
        
        
        #Visualize the image mapped to best components of one gaussian mixture model or the other
        #visualize(fgComponents)
        #visualize(bgComponents)
        
        #Now I need to make the graph
        g = igraph.Graph(directed=False)
        
        vertexList = [(y, x) for x in range(mask.shape[1]) for y in range(mask.shape[0])]
        g.add_vertices(map(str, vertexList))
        g.add_vertices(["source", "sink"])
    
        edgeList = []
        capacityList = []
        
        penaltyForCuttingSameComponent = 100
        
        #All horisontal edges
        print("Creating horisontal edges...")
        for x in range(mask.shape[1]):
            for y in range(mask.shape[0] - 1):
                pts = [(y, x), (y + 1, x)]
                edgeList.append(map(str, pts))
                
                if False:#bgComponents[pts[0]] == bgComponents[pts[1]] and fgComponents[pts[0]] == fgComponents[pts[1]]:
                    cap = penaltyForCuttingSameComponent
                else:
                    cap = binaryCostFunction(img[pts[0]], img[pts[1]])
                
                    
                capacityList.append(cap)
                
        #All vertical edges        
        print("Creating vertical edges...")
        for x in range(mask.shape[1] - 1):
            for y in range(mask.shape[0]):
                pts = [(y, x), (y, x + 1)]
                edgeList.append(map(str, pts))
                
                if False:#bgComponents[pts[0]] == bgComponents[pts[1]] and fgComponents[pts[0]] == fgComponents[pts[1]]:
                    cap = penaltyForCuttingSameComponent
                else:
                    cap = binaryCostFunction(img[pts[0]], img[pts[1]])
                    
                capacityList.append(cap)
        
        print("Binary edges range from " + str(min(capacityList)) + " to " + str(max(capacityList)) + ", with median of " + str(np.median(capacityList)) + " and average of " + str(np.average(capacityList)))
        
        #All edges to source and sink
        print("Creating edges to source and sink (unary term)...")
        unaryTerm = np.zeros(mask.shape)
        
        fgProb = fgGMM.score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)
        bgProb = bgGMM.score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)
        
        bias = -.0 #This is the bias towards the foreground (+) or backgrond (-)
        unaryTerm = fgProb - bgProb + bias

        k = max(capacityList)
        print("Setting k as: " + str(k))        

        unaryTerm[trimap == -1] = -k
        unaryTerm[trimap == 1] = k
        
        #visualize(unaryTerm)
        
        for x in range(mask.shape[1]):
            for y in range(mask.shape[0]):
                uterm = unaryTerm[y, x]
                
                if uterm > 0:
                    edgeList.append(("source", str((y,x))))
                    capacityList.append(uterm)
                elif uterm < 0:
                    edgeList.append((str((y,x)), "sink"))
                    capacityList.append(-uterm)
        
        print("Edge lists have been constructed")     
        print("We have : " + str(len(edgeList)) + " edges in our graph")        
        g.add_edges(edgeList)
        
        #igraph.plot(g.as_undirected(), layout="fr", vertex_label=None, edge_width=[2 * cap for cap in capacityList])
        
        assert len(edgeList) == len(capacityList)
        
        cuts = g.as_directed().all_st_mincuts("source", "sink", capacity=capacityList + capacityList)
        
        if len(cuts) == 0:
            print("No cuts found! Continuing onto the next iteration anyway....")
            continue
        #assert len(cuts) == 1, "%r cuts were found" % len(cuts)
        
        cut = cuts[0]
        
        print(cut)
        
        for vertexIndex in cut[0]:
            if vertexIndex < len(vertexList):
                tup = vertexList[vertexIndex]
                mask[tup] = 1
                
        for vertexIndex in cut[1]:
            if vertexIndex < len(vertexList):
                tup = vertexList[vertexIndex]
                mask[tup] = 0
                
        #visualize(mask)
        directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + ".bmp", mask)
    
    return mask

def binaryCostFunction(c1, c2):
    beta = .0001
    binaryEdgeWeight = 50
    return binaryEdgeWeight * math.exp(-beta * sum([x**2 for x in c1 - c2]))

def unaryCostFunction(gmm, pixel):
    prob = gmm.bic(np.asarray([pixel]))
    print(prob)
    #assert 0 <= prob <= 1
    return prob
    
def calcMaskUsingOpenCVGrabCut(img, bbox):
    mask = np.zeros(img.shape[:2],dtype='uint8')
    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))
    
    cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
    return mask

def main():
    print("Running GrabCut...")
    
    for img, bbox, filename in ImagesAndBBoxes:
        bbox = map(int, bbox)
        bbox = tuple(bbox)
        
        step = 10
        img = sp.misc.imresize(img, float(1)/step)
        bbox = [x/step for x in bbox]
        
        #mask = calcMaskUsingOpenCVGrabCut(img, bbox)
        mask = calcMaskUsingMyGrabCut(img, bbox, filename)
    
        #result = cv2.Image(mask)
        print("Finished one image.")
        directories.saveArrayAsImage(directories.output + filename + ".bmp", mask)
        
    print("GrabCut is finished :D")
    
if __name__ == "__main__":
    main()