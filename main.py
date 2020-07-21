import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras


def list_images(basePath, contains = None):
    # return the set of files that are valid
    return list_files(basePath, validExtentions = (".jpg", ".jpeg", ".png", ".bmp"), contains = contains)
