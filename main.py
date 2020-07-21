import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras


def list_images(basePath, contains = None):
    # return the set of files that are valid
    return list_files(basePath, validExtentions = (".jpg", ".jpeg", ".png", ".bmp"), contains = contains)

def list_files(basePath, validExtentions = (".jpg", ".jpeg", ".png", ".bmp"), contains = None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the string is not none and the filename does not contain the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
