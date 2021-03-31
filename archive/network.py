#Convolutional Neural Network

#Part 1 Building the CNN

#Importing the keras libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

#Intialising the CNN
classifier= Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3),activation='relu')) #(32-feature detector,3-rows,3-columns)
#input shape 64 is the pixel and 3 is for coloured images

#Step 2: Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Step 3: Flattening
classifier.add(Flatten())

#Step-4: Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=4, activation='softmax')) #use soft max for more than 1 output

#Compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])     #Use categorical_crossentropy for more labels

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train',
                                                    target_size=(32, 32),
                                                    batch_size=32,
                                                    class_mode='categorical')#use binary for 2 labels. for more labels google


test_set = test_datagen.flow_from_directory('test',
                                            target_size=(32, 32),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=2000,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=2000)

classifier.save('model.h5')