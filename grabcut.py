import numpy as np
import matplotlib.pyplot as plt
import cv2

import directories

ImagesAndBBoxes = directories.loadImagesAndBBoxes()

for img, bbox, filename in ImagesAndBBoxes:
    bbox = map(int, bbox)
    bbox = tuple(bbox)
    
    mask = np.zeros(img.shape[:2],dtype='uint8')
    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))
    
    cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
    """
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    """
    #result = cv2.Image(mask)
    mask *= float(255)/4
    cv2.cv.SaveImage(directories.output + filename + ".bmp", cv2.cv.fromarray(mask))
    