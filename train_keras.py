import os
os.environ.setdefault('PATH', '')
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Dense, AveragePooling2D, Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
from keras import regularizers 
from keras.losses import categorical_crossentropy
from keras import backend as K
import keras
import tensorflow as tf

weight_decay = 0.0001
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, kernel_size, padding='same', kernel_initializer='he_normal',
                name=conv_name_base + '2a',kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',kernel_initializer='he_normal', name=conv_name_base + '2b',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, kernel_size,padding='same', kernel_initializer='he_normal',strides=strides,
               name=conv_name_base + '2a',kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',kernel_initializer='he_normal',
               name=conv_name_base + '2b',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Conv2D(filters2, (1, 1), strides=strides,kernel_initializer='he_normal',
                      name=conv_name_base + '1',kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def bottleneck_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = Conv2D(filters1, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = layers.add([x, input_tensor])
    return x
    
image_size = 32
depth = 3
input_x = Input(shape=(image_size, image_size, depth))
x = Conv2D(16, (3, 3),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(input_x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

x = identity_block(x,(3,3),[16,16],stage=2,block='a')
x = identity_block(x,(3,3),[16,16],stage=2,block='b')
x = identity_block(x,(3,3),[16,16],stage=2,block='c')
x = identity_block(x,(3,3),[16,16],stage=2,block='e')
x = identity_block(x,(3,3),[16,16],stage=2,block='f')
x = identity_block(x,(3,3),[16,16],stage=2,block='g')
x = identity_block(x,(3,3),[16,16],stage=2,block='h')
x = identity_block(x,(3,3),[16,16],stage=2,block='j')
x = identity_block(x,(3,3),[16,16],stage=2,block='k')

x = conv_block(x,(3,3),[32,32],stage=3,block='a')
x = identity_block(x,(3,3),[32,32],stage=3,block='b')
x = identity_block(x,(3,3),[32,32],stage=3,block='c')
x = identity_block(x,(3,3),[32,32],stage=3,block='e')
x = identity_block(x,(3,3),[32,32],stage=3,block='f')
x = identity_block(x,(3,3),[32,32],stage=3,block='g')
x = identity_block(x,(3,3),[32,32],stage=3,block='h')
x = identity_block(x,(3,3),[32,32],stage=3,block='j')
x = identity_block(x,(3,3),[32,32],stage=3,block='k')

x = conv_block(x,(3,3),[64,64],stage=4,block='a')
x = identity_block(x,(3,3),[64,64],stage=4,block='b')
x = identity_block(x,(3,3),[64,64],stage=4,block='c')
x = identity_block(x,(3,3),[64,64],stage=4,block='e')
x = identity_block(x,(3,3),[64,64],stage=4,block='f')
x = identity_block(x,(3,3),[64,64],stage=4,block='g')
x = identity_block(x,(3,3),[64,64],stage=4,block='h')
x = identity_block(x,(3,3),[64,64],stage=4,block='j')
x = identity_block(x,(3,3),[64,64],stage=4,block='k')

x = AveragePooling2D((8, 8), name='avg_pool')(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)

model = Model(input_x, x)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])
######################################################################################
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.datasets import cifar10
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cifar10.load_data()
npad = ((0, 0), (4, 4), (4, 4), (0, 0))
x_train_raw_pad = np.pad(x_train_raw, pad_width=npad, mode='constant', constant_values=0)

train_X_1 = x_train_raw_pad[:, 0:image_size, 0:image_size, :]
train_y_1 = y_train_raw
x_train_raw = np.concatenate((train_X_1, x_train_raw), axis=0)
y_train_raw = np.concatenate((train_y_1, y_train_raw), axis=0)

train_X_2 = x_train_raw_pad[:, 0:image_size, -image_size:, :]
x_train_raw = np.concatenate((train_X_2, x_train_raw), axis=0)
y_train_raw = np.concatenate((train_y_1, y_train_raw), axis=0)

train_X_3 = x_train_raw_pad[:, -image_size:, 0:image_size, :]
train_y_3 = y_train_raw
x_train_raw = np.concatenate((train_X_3, x_train_raw), axis=0)
y_train_raw = np.concatenate((train_y_1, y_train_raw), axis=0)

train_X_4 = x_train_raw_pad[:, -image_size:, -image_size:, :]
train_y_4 = y_train_raw
x_train_raw = np.concatenate((train_X_4, x_train_raw), axis=0)
y_train_raw = np.concatenate((train_y_1, y_train_raw), axis=0)

flip_train_X = x_train_raw[:, :, ::-1, :]
flip_train_y = y_train_raw[:,:]
x_train_raw = np.concatenate((flip_train_X, x_train_raw), axis=0)
y_train_raw = np.concatenate((flip_train_y, y_train_raw), axis=0)


x_train = preprocess_input(x_train_raw.astype(np.float64))
y_train = np_utils.to_categorical(y_train_raw)

x_test = preprocess_input(x_test_raw.astype(np.float64))
y_test = np_utils.to_categorical(y_test_raw)

print()
print(x_train.shape)
print(y_train.shape)
print()
def list_rate_schedule(epoch):
    if epoch < 50:
        lr = 1e-3
    elif epoch < 75:
        lr = 1e-4
    else:
        lr = 1e-5
    return lr

lrsched = keras.callbacks.LearningRateScheduler(
    list_rate_schedule)

history = model.fit(x_train, y_train,  
                    batch_size=128,
                    epochs=100,     
                    verbose=1,         
                    callbacks=[lrsched],
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])