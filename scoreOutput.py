import numpy as np
import matplotlib.pyplot as plt
import cv2

import directories

def scoreOutput():
    output = directories.loadImagesInFolder(directories.output)
    groundTruth = directories.loadImagesInFolder(directories.groundTruth)
    
    for outputImage, groundTruthImage in zip(output, groundTruth):
        outputImage = np.asarray(outputImage[:-1])
        print(outputImage[-1])
        exit()
        groundTruthImage = np.asarray(groundTruthImage)
        
        print(np.count_nonzero(outputImage != groundTruthImage))
        
scoreOutput()