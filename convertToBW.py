from PIL import Image
import os

directories = [x[0] for x in os.walk('dataset') if x[0] != 'dataset']
print(directories)

image = Image.open('dataset/mammootty0new.jpg')
image = image.convert('1')
image.save('dataset/mammootty0new.jpg')

for peopleFolder in directories:
    listImages = [x for x in os.listdir(peopleFolder) if x.endswith('.jpg')]
    j = 0
    for image in listImages:
        # this is the RGB image array.
        imageFile = Image.open(peopleFolder + "/" + image)  # open color image
        imageFile = imageFile.convert('1')  # convert to BW
        imageFile.save(peopleFolder + "/" + peopleFolder[8:] + str(j) + "BW.jpg")
        j += 1