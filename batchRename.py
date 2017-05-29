#Script to batch rename database entries.


import os
directories = [x[0] for x in os.walk('dataset') if x[0] != 'dataset']
print(directories)

for peopleFolder in directories:
    listImages = [x for x in os.listdir(peopleFolder) if x.endswith('.jpg')]
    j = 0
    for image in listImages:
        # this is the RGB image array.
        os.rename(peopleFolder + "/" + image, peopleFolder + "/" + peopleFolder[8:] + str(j) + ".jpg")
        j += 1