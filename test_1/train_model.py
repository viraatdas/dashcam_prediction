import preprocessing as pre

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_data = pre.hsv

N_img_height = len(train_data)
N_img_width = len(train_data[0])
N_img_channels = 3


def nvidia_model():
    print ("Model is being trained")
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=inputShape))

    model.add(Convolution2D(24, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv1'))

    model.add(ELU())
    model.add(Convolution2D(36, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv2'))

    model.add(ELU())
    model.add(Convolution2D(48, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv4'))

    model.add(ELU())
    model.add(Convolution2D(64, (3, 3),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv5'))

    model.add(Flatten(name='flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())

    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')

    return model

validation_data = pre.validation
val_size = len(validation_data)
BATCH = 16
print('val_size: ', val_size)

filepath = 'model-weights-Vtest3.h5'
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=1,
                              verbose=1,
                              min_delta=0.23,
                              mode='min', )
modelCheckpoint = ModelCheckpoint(filepath,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1,
                                  save_weights_only=True)
callbacks_list = [modelCheckpoint]

model = nvidia_model()
history = model.fit_generator(
    train_data,
    steps_per_epoch=400,
    epochs=85,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=validation_data,
    validation_steps=val_size)

print(history)

