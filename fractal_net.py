import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Activation, merge, Lambda, Flatten, Dense
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD
from keras.engine.topology import Layer

from keras.callbacks import (
    Callback,
    LearningRateScheduler,
)
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

# constants

learning_rate = 0.01
momentum = 0.9
img_rows, img_cols = 32, 32
img_channels = 3
nb_epochs = 400
batch_size = 700
nb_classes = 10
pL = 0.5
weight_decay = 1e-4

(X_train, Y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
img_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)
img_gen.fit(X_train)
Y_train = np_utils.to_categorical(Y_train, nb_classes)

import traceback
class FractalLayer(Layer):
    def __init__(self, nb_filter, column=1, dim_ordering=K._BACKEND, pool=False, **kwargs):
        self.nb_filter = nb_filter
        self.column = column
        self.dim_ordering = dim_ordering
        self.gates = {}
        self.pool = pool
        # initializing gates to empty list
        for i in range(1, column+1):
            self.gates[i]=[]
        super(FractalLayer, self).__init__(**kwargs)
        
    def fc_block(self, z, nb_filter):
        fc = Convolution2D(nb_filter, 3, 3,
                            border_mode="same", W_regularizer=l2(weight_decay))(z)
        print "fc",z._keras_shape, nb_filter,fc._keras_shape, z, fc
        fc = BatchNormalization(axis=1)(fc)
        fc = Activation("relu")(fc)
        return fc

    def flush_gates(self, column):
        self.gates = {}
        # initializing gates to empty list
        for i in range(1, column+1):
            self.gates[i]=[]
        
    def basic_block(self, z, nb_filter, column, reset_gates=True):
        if reset_gates:
            self.flush_gates(column)
        fz = self.fc_block(z, nb_filter)
        if column >= 1:
            fc1 = self.basic_block(z, nb_filter, column-1, False)
            fc2 = self.basic_block(fc1, nb_filter, column-1, False)
            M1 = merge([fz,fc2], mode='ave')
            M1 = Activation("relu")(M1)
            gate = K.variable(1, dtype="uint8")
            self.gates[column].append(gate)
            return Lambda(lambda outputs: K.switch(gate, outputs[0], outputs[1]),
                          output_shape= lambda x: x[0])([fz, M1])
        else:
            return fz

    def call(self, inputs, mask=None):
        #import pdb; pdb.set_trace()
        #traceback.print_stack()
        if self.pool:
            inputs =  MaxPooling2D()(inputs)
            inputs = Activation("relu")(inputs)
        return self.basic_block(inputs, self.nb_filter, self.column)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'theano':
            rows = input_shape[2]
            cols = input_shape[3]
            print (input_shape[0], self.nb_filter, rows, cols)
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tensorflow':
            rows = input_shape[1]
            cols = input_shape[2]
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)


import numpy as np
class Gates_Callback(Callback):
    def __init__(self, gates):
        self.batch_count = 0
        self.gates = gates

    def on_batch_begin(self, batch, logs={}):
        if self.batch_count % 2 == 0:
            # Global regularization
            for depth in range(len(self.gates)):
                columns = len(self.gates[depth])+1
                selected_column = np.random.random_integers(low=1,high=columns)
                for i in range(1,columns):
                    if i >= selected_column:
                        for j in range(len(self.gates[depth][i])):
                            K.set_value(self.gates[depth][i][j], 1)
                    else:
                        for j in range(len(self.gates[depth][i])):
                            K.set_value(self.gates[depth][i][j], 0)
        else:
            # Local regularization
            for depth in range(len(self.gates)):
                columns = len(self.gates[depth])+1
                for i in range(1,columns):
                    for j in range(len(self.gates[depth][i])):
                        prob = np.random.uniform()
                        if prob > 0.5:
                            K.set_value(self.gates[depth][i][j], 1)
                        else:
                            K.set_value(self.gates[depth][i][j], 0)
        self.batch_count = self.batch_count+1

    def on_train_end(self, logs={}):
        for i in gates:
            K.set_value(gates[i][1],1)
            
def scheduler(epoch):
    if epoch < nb_epochs/2:
        return learning_rate
    elif epoch < nb_epochs*3/4:
        return learning_rate*0.1

inputs = Input(shape=(img_channels, img_rows, img_cols))
layer_1 = FractalLayer(64,2)
predictions = layer_1(inputs)

layer_2 = FractalLayer(128,3,pool=True)
predictions = layer_2(predictions)

layer_3 = FractalLayer(256,4,pool=True)
predictions = layer_3(predictions)

# predictions = basic_block(inputs, 32, 3)

flatten1 = Flatten()(predictions)
predictions = Dense(output_dim=10, init="he_normal", activation="softmax", W_regularizer=l2(weight_decay))(flatten1)
model = Model(input=inputs, output=predictions)
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

gates = [layer_1.gates, layer_2.gates, layer_3.gates]

model.fit_generator(img_gen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                    samples_per_epoch=len(X_train),
                    nb_epoch=nb_epochs,
                    callbacks=[Gates_Callback(gates), LearningRateScheduler(scheduler)])