import PIL
from scipy import misc
import os
import scipy.misc
from PIL import Image


# traverse through images in dataset.
directories = [x[0] for x in os.walk('dataset') if x[0] != 'dataset']
# print(directories)

for peopleFolder in directories:
    listImages = [x for x in os.listdir(peopleFolder) if not x.endswith('BW.jpg')]
    for image in listImages:
        # this is the RGB image array.
        # print(peopleFolder+"/"+image)
        os.remove(peopleFolder+"/"+image)



