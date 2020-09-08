# Tutorial from: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
import numpy as np
# np.random.seed(123)  # for reproducibility

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist

# from read_activations import get_activations, display_activations
from read_activations_image import get_activations, display_activations

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print (X_train.shape)
# (60000, 28, 28)

from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

# Reshape the data so that it fits the tensor requirement of the CNN.
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print (X_train.shape)

# Convert data type and normalize valuesPython
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)
print(y_train[:10])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print( Y_train.shape )
# (60000, 10)
print(Y_train[:10])

def createModel():
    model = Sequential()
    # model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28)))  # Need to reverse input shape.
    model.add(Convolution2D(4, 3, 3, activation='relu', input_shape=(28, 28, 1)))  # Need to reverse input shape.
    print(model.output_shape)
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(4, 3, 3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(4, 3, 3, activation='relu'))
    model.add(Convolution2D(4, 3, 3, activation='relu'))
    model.add(Convolution2D(4, 3, 3, activation='relu'))

    model.add(Dropout(0.25))
    print( model.output_shape )

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    print(model.output_shape)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def runTest(trainingSize=60000, epochs=1):
    print( "Creating model" )
    model = createModel()

    # trainingSize = 10000
    # epochs = 1
    verbosity = 2
    model.fit(X_train[:trainingSize], Y_train[:trainingSize],
              batch_size=32, nb_epoch=epochs, verbose=verbosity)


    print('Evaluating score')
    score = model.evaluate(X_test, Y_test, verbose=verbosity)
    print('Loss:', str(score[0]), ' Accuracy:', str(score[1]))

    print("XTest Shape:", X_test[0:1].shape)
    randomInput = np.random.random_sample( X_test[0:1].shape) * np.max(X_test[0:1])
    # print(randomInput)
    a = get_activations(model, X_test[0:1], print_shape_only=True)  # with just one sample.
    # a = get_activations(model, randomInput, print_shape_only=True)  # with random sample.
    display_activations(a)

TEST=False

if not TEST:
    runTest(60000, 2)
#plt.imshow(X_test[0])
else:
    plt.imshow(np.reshape(X_test[0], (28,28)), cmap='gray')
plt.show()