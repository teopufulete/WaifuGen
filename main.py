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
                
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            
            # check to see if the file is an image and needs processing
            if ext.endswith(validExtentions):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath

def load_images(directory = '../venv/data/data', size = (64,64)):
    images, labels = [], []
    label = 0

    imagePaths = list(list_images(directory))
    for path in imagePaths:
        if not ('OSX' in path):
            path = path.replace('\\', '/')
            image = cv2.imread(path)  # Reading the image with OpenCV
            image = cv2.resize(image, size)  # Resizing the image, in case some are not of the same size
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images

images = load_images('../venv/data/data')


_,ax = plt.subplots(5,5, figsize = (8,8))
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(images[5*i+j])
        ax[i,j].axis('off')
        
        
        
class GAN():
    def __init__(self):
        self.img_shape = (64, 64, 3)
        self.noise_size = 100
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy',
                                   optimizer = optimizer,
                                   metrics = ['accuracy'])
        self.discriminator.trainable = False
