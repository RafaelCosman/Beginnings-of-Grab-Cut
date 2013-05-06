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
    
def calcMaskUsingMine(img, bbox):
    print(bbox)
    print(img.shape)
    
    trimap = np.ones(img.shape[:2],dtype='int8')
    trimap *= -1
    
    for x in range(bbox[0], bbox[2]):
        for y in range(bbox[1], bbox[3]):
            trimap[y, x] = 0
    
    mask = np.copy(trimap)
    mask += 1
    
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
    
    #Make FG Components
    #gmm = sklearn.mixture.GMM(n_components=5, covariance_type='full', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc')
    fgGMM = sklearn.mixture.DPGMM(n_components=5, alpha=.001, covariance_type='full', random_state=None, thresh=0.01, min_covar=0.001, n_iter=1, params='wmc', init_params='wmc')
    fgGMM.fit(fgObs)
    
    print(np.round(fgGMM.weights_, 2))
    print(np.round(fgGMM.means_, 2))
    
    fgComponents = fgGMM.predict(np.reshape(img, (-1, 3)))
    print(len(fgComponents))
    print(fgComponents.shape)
    fgComponents = np.reshape(fgComponents, mask.shape)
    
    
    #Make BG Components
    #gmm = sklearn.mixture.GMM(n_components=5, covariance_type='full', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc')
    bgGMM = sklearn.mixture.DPGMM(n_components=5, alpha=.001, covariance_type='full', random_state=None, thresh=0.01, min_covar=0.001, n_iter=1, params='wmc', init_params='wmc')
    bgGMM.fit(bgObs)
    
    print(np.round(bgGMM.weights_, 2))
    print(np.round(bgGMM.means_, 2))
    
    bgComponents = bgGMM.predict(np.reshape(img, (-1, 3)))
    print(len(bgComponents))
    bgComponents = np.reshape(bgComponents, mask.shape)
    
    
    #visualize(components)
    
    #Now I need to make the graph
    g = igraph.Graph(directed=True)
    
    g.add_vertices([str((y, x)) for x in range(mask.shape[1]) for y in range(mask.shape[0])])
    g.add_vertices(["source", "sink"])

    edgeList = []
    capacityList = []
    
    #All horisontal edges
    print("shape: " + str(mask.shape))
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0] - 1):
            pts = (str((y, x)), str((y + 1, x)))
            
            edgeList.append(pts)
            edgeList.append(pts[::-1])
            
            print((y, x))
            capacity = binaryCostFunction(img[y, x], img[y + 1, x])
            
            capacityList.append(capacity)
            capacityList.append(capacity)
    
    #All vertical edges
    for x in range(mask.shape[1] - 1):
        for y in range(mask.shape[0]):
            pts = (str((y, x)), str((y, x + 1)))
            
            edgeList.append(pts)
            edgeList.append(pts[::-1])
            
            capacity = binaryCostFunction(img[y, x], img[y, x + 1])
            
            capacityList.append(capacity)
            capacityList.append(capacity)
            
    k = 100
    
    #All edges to source and sink
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            tmap = trimap[y, x]
            
            if tmap == -1: #Background
                edgeList.append((str((y,x)), "sink"))
                capacityList.append(k)
            
            elif tmap == 1: #Foreground
                edgeList.append(("source", str((y,x))))
                capacityList.append(k)

            elif tmap == 0: #Unknown
                pixel = img[y, x]
                
                edgeList.append(("source", str((y,x))))
                capacityList.append(fgGMM.aic([pixel]))
                
                edgeList.append((str((y,x)), "sink"))
                capacityList.append(bgGMM.aic([pixel]))

            else:
                print("ERR: unexpected value " + str(tmap) + " found in trimap")
                exit()
    
    g.add_edges(edgeList)
    
    print("We have : " + str(len(edgeList)) + " edges in our graph")
    
    igraph.plot(g, layout="fr", vertex_label=None)
    
    cuts = g.all_st_mincuts("source", "sink", capacity=capacityList)
    
    for cut in cuts:
        print(cut)
    
    exit()
    
    return mask

def binaryCostFunction(c1, c2):
    beta = .0001
    return math.exp(beta * sum([x**2 for x in c1 - c2]))    
    
def calcMaskUsingOpenCV(img, bbox):
    mask = np.zeros(img.shape[:2],dtype='uint8')
    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))
    
    cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
    return mask

for img, bbox, filename in ImagesAndBBoxes[1:]:
    bbox = map(int, bbox)
    bbox = tuple(bbox)
    
    step = 100
    img = img[::step, ::step]
    bbox = [x/step for x in bbox]
    
    #mask = calcMaskUsingOpenCV(img, bbox)
    mask = calcMaskUsingMine(img, bbox)

    #result = cv2.Image(mask)
    mask *= float(255)/4
    print("Finished one image.")
    cv2.cv.SaveImage(directories.output + filename + ".bmp", cv2.cv.fromarray(mask))