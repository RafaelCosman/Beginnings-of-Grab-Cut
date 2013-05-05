import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import sklearn
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import directories
from visualization import *

ImagesAndBBoxes = directories.loadImagesAndBBoxes()
directories.ensure_dir(directories.output)
    
def calcMaskUsingMine(img, bbox):
    print(bbox)
    
    mask = np.zeros(img.shape[:2],dtype='uint8')
    
    for x in range(bbox[0], bbox[2]):
        for y in range(bbox[1], bbox[3]):
            mask[y, x] = 1
            
    fgObs = img[mask == 1]
    print(fgObs.shape)
          
    bgObs = img[mask == 0]
    print(bgObs.shape)

    allObs = np.asarray(list(fgObs) + list(bgObs))
    print(allObs.shape)
    
    #plt.scatter(x=allObs[::100, 2], y=allObs[::100, 0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    step = 10000
    """
    colorList = [0] * len(fgObs) + [1] * len(bgObs)
    colorArr = np.zeros((len(allObs), 3))
    
    colorArr[:, 0] = 255 *  np.asarray(colorList)
    colorArr[:, 1] = 255 * (1 - np.asarray(colorList))
    colorArr[:, 2] = 255
    
    ax.scatter(allObs[::step, 0], allObs[::step, 1], allObs[::step, 2], c=colorArr)
    
    plt.show()   
    exit()
    """
    #gmm = sklearn.mixture.GMM(n_components=5, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc')
    gmm = sklearn.mixture.DPGMM(n_components=5, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=1, params='wmc', init_params='wmc')
    
    gmm.fit(fgObs)
    
    print(np.round(gmm.weights_, 2))
    print(np.round(gmm.means_, 2))
    
    components = gmm.predict(np.reshape(img, (-1, 3)))
    print(len(components))
    print(components.shape)
    components = np.reshape(components, mask.shape)
    
    visualize(components)
    
    print("Done")
    exit()
    
def calcMaskUsingOpenCV(img, bbox):
    mask = np.zeros(img.shape[:2],dtype='uint8')
    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))
    
    cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
    return mask

for img, bbox, filename in ImagesAndBBoxes:
    bbox = map(int, bbox)
    bbox = tuple(bbox)
    
    #mask = calcMaskUsingOpenCV(img, bbox)
    mask = calcMaskUsingMine(img, bbox)

    #result = cv2.Image(mask)
    mask *= float(255)/4
    print("Finished one image.")
    cv2.cv.SaveImage(directories.output + filename + ".bmp", cv2.cv.fromarray(mask))
    