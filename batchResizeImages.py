import PIL
from scipy import misc
import os
import scipy.misc
from PIL import Image


# firstDqFace = misc.imread('dataset/dqFirst.jpg')
# firstDqFace = scipy.misc.imresize(firstDqFace, (64, 64))
# misc.imsave('dataset/dqFirstnew.jpg', firstDqFace)

# traverse through images in dataset.
directories = [x[0] for x in os.walk('dataset') if x[0] != 'dataset']
# print(directories)

for peopleFolder in directories:
    listImages = [x for x in os.listdir(peopleFolder) if x.endswith('.jpg')]
    j = 0
    for image in listImages:
        # this is the RGB image array.
        face = misc.imread(peopleFolder + '/' + image)

        #resize to 64x64
        i_width = 64
        i_height = 64
        face = scipy.misc.imresize(face, (i_height, i_width))

        #save image.
        misc.imsave(peopleFolder + "/" + peopleFolder[8:] + str(j) + "new.jpg", face)
        j += 1




