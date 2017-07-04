import os
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, ELU
from keras.layers import Dropout, MaxPooling2D, Activation, AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import augment
import random

#source activate carnd-term1
#if cv2 fails.. 
#pip install opencv-python
steering_scale = 25.0

def generator(samples, batch_size=32, perc_to_augment=0.5):
    num_samples = len(samples)
    do_augment = True
    if do_augment:
        shadows = augment.load_shadow_images('./shadows/*.png')    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                fullpath, steering = batch_sample

                image = Image.open(fullpath)
                if image is None:
                    continue

                #PIL Image as a numpy array
                image = np.array(image)
                if do_augment and random.uniform(0.0, 1.0) < perc_to_augment:
                    image = augment.augment_image(image, shadows)
                center_angle = steering * steering_scale
                images.append(image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


def read_log_csv(filename):
    lines = []
    with open(filename) as infile:
        reader = csv.DictReader(infile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def show_model_summary(model):
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)

def make_model():
    model = Sequential()

    input_shape=(160,320,3)

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    #use half the image data in each dimension
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #use half the image data in each dimension
    model.add(AveragePooling2D(pool_size=(2, 2)))

    #normalize color data to center on zero with mean variance of 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    #model.add(Lambda(lambda x: x/127.5 - 1.,
    #        input_shape=input_shape,
    #        output_shape=input_shape))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(640))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Activation('tanh'))

    #a single float output for steering command
    model.add(Dense(1))

    #choose a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model

def make_model_four():
    model = Sequential()

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
    
    #use half the image data in each dimension
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #normalize color data to center on zero with mean variance of 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(640))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())

    #a single float output for steering command
    model.add(Dense(1))

    #choose a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model

def make_model_three():
    model = Sequential()

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    #use half the image data in each dimension
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #normalize color data to center on zero with mean variance of 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1920))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(500))
    model.add(ELU())
    
    # model.add(Dense(100))
    # model.add(ELU())
    # model.add(Dense(50))
    # model.add(ELU())
    # model.add(Dense(10))
    # model.add(ELU())

    #a single float output for steering command
    model.add(Dense(1))

    #choose a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model


def make_model_two():
    model = Sequential()

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

    #normalize color data to center on zero with mean variance of 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1920))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(ELU())

    #a single float output for steering command
    model.add(Dense(1))

    #choose a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model

def load_csv(filename, data_path):
    ret = []
    filename = os.path.join(data_path, filename)

    lines = read_log_csv(filename)
    for line in lines:
        source_path = line['center']
        filename = os.path.basename(source_path)
        fullpath = os.path.join(data_path, 'IMG', filename)
        if not os.path.exists(fullpath):
            continue
        steering = float(line['steering'])
        ret.append((fullpath, steering))
    return ret

def load_csvs(filename, data_paths):
    ret = []
    for path in data_paths:
        ret = ret + load_csv(filename, path)
    return ret

def load_data(filename, data_paths):
    images = []
    measurements = []
    lines = load_csvs(filename, data_paths)
    for line in lines:
        fullpath, steering = line
        #cv2 was reading this as a BGR or something
        #image = cv2.imread(fullpath)
        image = Image.open(fullpath)
        if image is None:
            continue

        #PIL Image as a numpy array
        image = np.array(image)

        images.append(image)
        measurements.append(steering)

    return np.array(images), np.array(measurements)

def make_generators(filename, data_paths, batch_size=32, cvs_val_filename=None):
    lines = load_csvs(filename, data_paths)
    #if cvs_val_filename is not None and os.path.exists(cvs_val_filename):
    #    train_samples = lines
    #    validation_samples = load_csvs(cvs_val_filename, data_paths)
    #else:
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size, perc_to_augment=0.5)
    validation_generator = generator(validation_samples, batch_size=batch_size, perc_to_augment=0.0)
    n_train = len(train_samples)
    n_val = len(validation_samples)
    return train_generator, validation_generator, n_train, n_val


def train():
    data_paths = ['./more_driving', './tawn_drive', './DriveData']
    #data_path = 'd:/projects/udacity_car_sim/log/'
    #data_path = 'f:/log/'
    #data_path = 'c:/temp/log/'
    #data_path = 'd:/projects/udacity_car_sim/tawn_drive/'
    #data_path = 'd:/projects/udacity_car_sim/data/'
    cvs_filename = 'driving_log.csv'
    cvs_val_filename = 'validation_log.csv'
    use_generators = True
    epochs = 50
    batch_size = 128

    if use_generators:
        train_generator, validation_generator, n_train, n_val = make_generators(cvs_filename, data_paths, batch_size, cvs_val_filename)
    else:
        X_train, y_train = load_data(cvs_filename, data_paths)


    model = make_model()

    show_model_summary(model)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, verbose=0),
        ModelCheckpoint("model_best.h5", monitor='val_loss', save_best_only=True, verbose=0),
    ]

    if use_generators:
        history_object = model.fit_generator(train_generator, 
            samples_per_epoch = n_train,
            validation_data = validation_generator,
            nb_val_samples = n_val,
            nb_epoch=epochs,
            verbose=1,
            callbacks=callbacks)
    else:
        history_object = model.fit(X_train, y_train, validation_split=0.2,
            shuffle=True, nb_epoch=epochs,
            callbacks=callbacks)

    model.save('model.h5')
    print('training done.')


    
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

train()
