import cv2
import sys
import os

data = "data_GT/"
bboxes = "ICCV09_new_bounding_boxes/"
groundTruth = "seg_GT/"
output = "output/"

bboxSuffix = ".txt"

def loadImageByName(name):
    filename = data + name
    img = cv2.imread(filename)
    return img

def loadBBoxByName(name):    
    filename = bboxes + name + bboxSuffix
    
    with open(filename) as f:
        return map(float, f)
    
def loadImagesAndBBoxes():
    filenames = os.listdir(data)
    images = map(loadImageByName, filenames)
    
    assert not None in images
    
    filenames = [fname.split(".")[0] for fname in filenames]
    bboxes = map(loadBBoxByName, filenames)
    
    assert not None in bboxes
    
    return zip(images, bboxes, filenames)