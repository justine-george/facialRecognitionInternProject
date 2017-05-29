from PIL import Image
from keras.models import model_from_json
import numpy
import os
import face_extraction as getFace
from scipy import misc
import scipy.misc


# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

# load data
X_train = getFace.faceRepo
y_train = getFace.faceLabelRepo
list = getFace.listOfCelebrities


X_test = []
y_test = []

i = 0
while i < 199:
    X_test.append(X_train[i])
    y_test.append(y_train[i])
    i += 25

X_test = numpy.array(X_test)
y_test = numpy.array(y_test)


# load image from toBePredicted folder.
listImages = [x for x in os.listdir('toBePredicted') if x.endswith('.jpg')]

for images in listImages:
    newFace = misc.imread('toBePredicted/' + images)
    newFace = numpy.array([newFace])
    prediction = loaded_model.predict(newFace)
    pred = numpy.argmax(prediction)
    # print name of the celebrity, waiting to get predicted :P
    for x in range(10,18):
        if x == pred:
            print("Predicted class - ", list[x-10], "\nimage name:",  images,"\n-------------")
            break

# # checking with test values.
# j = 0
# correct = 0
# total = 8
# for image in X_test:
#     img = numpy.array([image])
#     print("Original - ", list[j])
#     prediction = loaded_model.predict(img)
#     pred = numpy.argmax(prediction)
#     for x in range(10, 18):
#         if x == pred:
#             print("Predicted class - ", list[x - 10])
#             if list[j] == list[x-10]:
#                 correct += 1
#             break
#     print("---")
#     j += 1
#
# # show percentage accuracy.
# print("Test accuracy:")
# scoreBoard = correct*100/total
# print(scoreBoard, "% accurate.")