# load dataset to numpy arrays, later to be trained.
import numpy as np
import os
from PIL import Image
from scipy import misc
from skimage import io

# initialise a 4D array.
faceRepo = misc.imread('dataset/mammootty0new.jpg')
faceRepo = np.array([faceRepo])

# initialise label array. (supervised learning).
faceLabelRepo = np.array([10]) # 10 for mammootty

# traverse through images in dataset.
directories = [x[0] for x in os.walk('dataset') if x[0] != 'dataset']
print(directories)

# list of celebrities in training set.
listOfCelebrities = []

for peopleFolder in directories:
    listOfCelebrities.append(peopleFolder[8:])

    listImages = [x for x in os.listdir(peopleFolder) if x.endswith('.jpg')]
    for image in listImages:
        # this is the RGB image array.
        face = misc.imread(peopleFolder + '/' + image)
        faceRepo = np.append(faceRepo, [face],axis=0)
        #assigning label
        bit = peopleFolder[8:]   # peoplefolder = "dataset/<fodler_name>" , folder name starting from index 8.
        if bit == 'dulquer':
            label = int(13)
        elif bit == 'jyothika':
            label = int(17)
        elif bit == 'mammootty':
            label = int(10)
        elif bit == 'mohanlal':
            label = int(12)
        elif bit == 'nivin':
            label = int(11)
        elif bit == 'shobana':
            label = int(15)
        elif bit == 'suriya':
            label = int(14)
        elif bit == 'vijay':
            label = int(16)
        else:
            pass
        faceLabelRepo = np.append(faceLabelRepo, [label],axis=0)
