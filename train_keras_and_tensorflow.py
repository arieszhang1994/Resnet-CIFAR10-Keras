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
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
K.set_learning_phase(1)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=False # True means not use the full gpu memory
    )
)
set_session(tf.Session(config=config))

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

x_input = tf.placeholder(tf.float32, [None, image_size, image_size, depth])
y = model(x_input)
t = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32, [])
cost = tf.reduce_mean(categorical_crossentropy(t, y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
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

sess = tf.Session(config=config)
n_epochs = 100
batch_size = 128
n_batches = x_train.shape[0] // batch_size
init = tf.global_variables_initializer()
sess.run(init)

valid_best = 1000
for epoch in range(n_epochs):
    train_cost = 0
    x_train, y_train = shuffle(x_train, y_train, random_state=43)
    if epoch < 50:
        lr = 1e-3
    elif epoch < 75:
        lr = 1e-4
    else:
        lr = 1e-5
    K.set_learning_phase(1)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        _, _cost = sess.run([train, cost], feed_dict={x_input: x_train[start: end], 
                                                        t: y_train[start: end], 
                                                        learning_rate:lr})
        train_cost += _cost
    # vert important for evaluting    
    K.set_learning_phase(0)
    valid_cost, pred_y = sess.run([cost, y], feed_dict={x_input: x_test, 
                                                        t: y_test})

    if valid_cost < valid_best:
        model.save('weights/resnet.h5')
        valid_best = valid_cost
    print('EPOCH:{0:d}, train_cost:{1:.5f}, valid_cost:{2:.5f},accuracy_score:{3:.5f}' .format(
        epoch + 1, train_cost*batch_size/x_train.shape[0], valid_cost,
        accuracy_score(np.argmax(y_test,1),np.argmax(pred_y,1))))
