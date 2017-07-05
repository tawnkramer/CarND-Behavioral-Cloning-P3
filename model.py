'''
    File: model.py
    Author : Tawn Kramer
    Date : July 2017
'''
import os
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, ELU
from keras.layers import Dropout, MaxPooling2D, Activation, AveragePooling2D
from keras.layers import Conv2D, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU
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

'''
Steering is from -1. to 1. so we apply a scale so that error values make more sense.
'''
steering_scale = 25.0


def generator(samples, batch_size=32, perc_to_augment=0.5):
    '''
    Rather than keep all data in memory, we will make a function that keeps
    it's state and returns just the latest batch required via the yield command.
    
    As we load images, we can optionally augment them in some manner that doesn't
    change their underlying meaning or features. This is a combination of
    brightness, contrast, sharpness, and color PIL image filters applied with random
    settings. Optionally a shadow image may be overlayed with some random rotation and
    opacity.

    We flip each image horizontally and supply it as a another sample with the steering
    negated.
    '''
    num_samples = len(samples)
    do_augment = True
    if do_augment:
        shadows = augment.load_shadow_images('./shadows/*.png')    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        #divide batch_size in half, because we double each output by flipping image.
        batch_size = int(batch_size / 2)
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

                #flip image and steering.
                image = np.fliplr(image)
                center_angle = -center_angle
                images.append(image)
                angles.append(center_angle)


            # final np array to submit to training
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


def show_model_summary(model):
    '''
    show the model layer details
    '''
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)



def make_model():
    '''
    Make a Keras network model. This architecture uses the 5 convolutional layers
    inspired by Nidia's seminal paper: End to End Learning for Self-Driving Cars 
    https://arxiv.org/abs/1604.07316 
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    '''
    model = Sequential()

    input_shape=(160,320,3)

    # set up cropping2D layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

    #normalize color data to center on zero with mean variance of 1
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

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
    model.add(Dense(1920))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('tanh'))

    #a single float output for steering command
    model.add(Dense(1))

    #choose a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model


def read_log_csv(filename):
    '''
    read lines of CSV file
    '''
    lines = []
    with open(filename) as infile:
        reader = csv.DictReader(infile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def load_csv(filename, data_path):
    '''
    get the center and steering data from the csv files
    '''
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
    '''
    iterate over multiple paths to get csv data
    '''
    ret = []
    for path in data_paths:
        ret = ret + load_csv(filename, path)
    return ret



def make_generators(filename, data_paths, batch_size=32):
    '''
    load the job spec from the csv and create some generator for training
    '''
    
    #get the image/steering pairs from the csv files
    lines = load_csvs(filename, data_paths)
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size, perc_to_augment=0.0)
    validation_generator = generator(validation_samples, batch_size=batch_size, perc_to_augment=0.0)
    
    #double each because we will flip image in generator
    n_train = len(train_samples) * 2
    n_val = len(validation_samples) * 2
    
    return train_generator, validation_generator, n_train, n_val



def train():
    '''
    Use Keras to train an artificial neural network to use end-to-end behavorial cloning to drive a vehicle.
    '''
    data_paths = ['./tawn_drive', './more_driving', './left_turn', './DriveData', './challenge']
    cvs_filename = 'driving_log.csv'
    epochs = 10
    batch_size = 128

    train_generator, validation_generator, n_train, n_val = make_generators(cvs_filename, data_paths, batch_size)

    model = make_model()

    show_model_summary(model)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, verbose=0),
        ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(train_generator, 
        samples_per_epoch = n_train,
        validation_data = validation_generator,
        nb_val_samples = n_val,
        nb_epoch=epochs,
        verbose=1,
        callbacks=callbacks)

    print('training complete.')
    
    ### print the keys contained in the history object
    print(history.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

train()
