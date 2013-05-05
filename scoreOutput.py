import numpy as np
import matplotlib.pyplot as plt
import cv2

import directories 

def greyscale(img):
    return np.average(img, 2)

def threshold(arr, threshold):
    arr[arr<threshold] = 0
    arr[arr != 0] = 255

def scoreOutput():
    output = directories.loadImagesInFolder(directories.output)
    groundTruth = directories.loadImagesInFolder(directories.groundTruth)
    
    for outputImage, groundTruthImage in zip(output, groundTruth):
        fname = outputImage[1]
        outputImage = np.asarray(outputImage)[0]
        outputImage = greyscale(outputImage)
        threshold(outputImage, 150)
        
        groundTruthImage = np.asarray(groundTruthImage)[0]
        groundTruthImage = greyscale(groundTruthImage)
        
        pixelsDifferent = np.count_nonzero(outputImage != groundTruthImage)
        pixelsTotal = outputImage.shape[0] * outputImage.shape[1]
        """
        visualize(outputImage)
        visualize(groundTruthImage)
        """
        print fname + ":   \t" + "{:.0%}".format(float(pixelsDifferent)/pixelsTotal)
        
scoreOutput()