import cv2
import sys
import os

data = "data_GT/"
bboxes = "ICCV09_new_bounding_boxes/"
groundTruth = "seg_GT/"
output = "output/"

bboxSuffix = ".txt"

def loadImageByName(filepath):
    return cv2.imread(filepath)

def loadBBoxByName(name):    
    filename = bboxes + name + bboxSuffix
    
    with open(filename) as f:
        return map(float, f)
    
def loadImagesAndBBoxes():
    x = loadImagesInFolder(data)
    
    images = [tup[0] for tup in x]
    filenames = [tup[1] for tup in x]
    
    bboxes = map(loadBBoxByName, filenames)
    
    assert not None in bboxes
    
    return zip(images, bboxes, filenames)

def loadImagesInFolder(folder):
    filenames = os.listdir(folder)
    images = [loadImageByName(folder + filename) for filename in filenames]
    
    assert not None in images
    
    filenames = [fname.split(".")[0] for fname in filenames]
    
    return zip(images, filenames)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)