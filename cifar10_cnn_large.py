#facial recognition

import numpy
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import h5py
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('tf')

# user generated script to extract dataset.
import face_extraction as getFace

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load data
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()     36 index

# load data
X_train = getFace.faceRepo
y_train = getFace.faceLabelRepo

X_test = []
y_test = []

# getting data for testing, equal representation from all 8 categories ensured.
i = 0
while i < 199:
    X_test.append(X_train[i])
    y_test.append(y_train[i])
    i += 25

X_test = numpy.array(X_test)
y_test = numpy.array(y_test)

# print(X_test.shape)
# print(y_test.shape)
# print(y_test)

# # # split to test and training data.
# # X_train, X_test = train_test_split(trainX, test_size=0.1)
# # y_train, y_test = train_test_split(trainY, test_size=0.1)
# #
# # print('X train - ', X_train.shape)
# # print('y train - ', y_train.shape)
# # print('x test - ', X_test.shape)
# # print('y test - ', y_test.shape)
#
#
#
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]



# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# Compile model - default values
# epochs = 25
# lrate = 0.01

epochs = 100
lrate = 0.01

decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

# Fit the model
model.fit(X_train, y_train, nb_epoch=epochs, batch_size=32)


# # Fit the model
# model.fit(X_train, y_train, nb_epoch=epochs, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# serialize model to JSONx
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")