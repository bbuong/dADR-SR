#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a dilated Adaptive Residual Network for multiscale super-resolution of ocean Absolute Dynamic Topography based on multivariate predictors
Reference: doi:........
@author: Bruno Buongiorno Nardelli
Consiglio Nazionale delle Ricerche
Istituto di Scienze Marine
Napoli, Italia
"""


import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # specify to ignore warning messages

import tensorflow as tf
from keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation, Add, Multiply, Reshape, Concatenate
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers.core import Lambda
from tensorflow.python.keras import backend as K
import scipy.io as sio
from netCDF4 import Dataset
from netCDF4 import getlibversion
import glob
from tensorflow.keras.models import load_model

from DropBlock2D import DropBlock2D

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def read_nc(netcdf_file):
    ncid = Dataset(netcdf_file, 'r')

    nc_vars = [var for var in ncid.variables]
    for var in nc_vars:
        if hasattr(ncid.variables[str(var)], 'add_offset'):
            exec('global ' + str(var) + "; offset=ncid.variables['" + str(var) + "'].add_offset; " + str(
                var) + "=ncid.variables['" + str(var) + "'][:]-offset")
        else:
            exec('global ' + str(var) + '; ' + str(var) + "=ncid.variables['" + str(var) + "'][:]")
    ncid.close()
    return



####################################
#CNN model configuration parameters
####################################

pat=5
n_epochs = 150
val_split=.15
keep_prob=.9

##################################################################
#
#   model definition/fit
#
##################################################################

model_dir='trained_models/'
model_name=model_dir+'dilated_drop_ADR-SR_CNN_ADT_MODEL_cuda.h5'


if not glob.glob(model_name):
    
    def se_block(tensor, ratio=10):
        
        channel_axis = 1    
        nb_channels = tensor.shape[channel_axis]
        x_shape = (nb_channels, 1, 1)
        
        x = GlobalAveragePooling2D(data_format="channels_first")(tensor)
        x = Dense(nb_channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
        x = Dense(nb_channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
        x = Reshape(x_shape)(x)
        
        x = Multiply()([tensor, x])
        return x

    def se_res_block_dilated_drop(input_tensor, n_id_block, block_n, filters):
        
        x_se1 = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, strides=1, padding='same', data_format="channels_first")(input_tensor)
        x_se1 = Activation('relu')(x_se1)
        x_se1 = Conv2D(filters=10, kernel_size=3, dilation_rate=1,strides=1, padding='same', data_format="channels_first")(x_se1)
        
        x_se2 = Conv2D(filters=filters, kernel_size=3, dilation_rate=3, strides=1, padding='same', data_format="channels_first")(input_tensor)
        x_se2 = Activation('relu')(x_se2)
        x_se2 = Conv2D(filters=10, kernel_size=3, dilation_rate=3,strides=1, padding='same', data_format="channels_first")(x_se2)
        
        x_se3 = Conv2D(filters=filters, kernel_size=3, dilation_rate=5, strides=1, padding='same', data_format="channels_first")(input_tensor)
        x_se3 = Activation('relu')(x_se3)
        x_se3 = Conv2D(filters=10, kernel_size=3, dilation_rate=5,strides=1, padding='same', data_format="channels_first")(x_se3)
        
        x = Concatenate(axis=1)([x_se1, x_se2, x_se3])

        x = se_block(x)  
        x = Add()([x, input_tensor])
        if block_n>n_id_block-2:
            x = DropBlock2D(block_size=7, keep_prob=keep_prob, data_format="channels_first",sync_channels='true')(x)
        
        return x
    
    # def se_res_block(input_tensor, filters):
    #     x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', data_format="channels_first")(input_tensor)
    #     x = Activation('relu')(x)

    #     x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', data_format="channels_first")(x)
    #     x = se_block(x)  
    #     x = Add()([x, input_tensor])

    #     return x
    

    def dadrsr(filters=120, n_id_block=12):
        
        inputs = Input(shape=(4,76, 100))

        x_1a = Conv2D(filters=10, kernel_size=3, dilation_rate=1, strides=1, padding='same', data_format="channels_first")(inputs)
        x_1b = Conv2D(filters=10, kernel_size=3, dilation_rate=3, strides=1, padding='same', data_format="channels_first")(inputs)
        x_1c = Conv2D(filters=10, kernel_size=3, dilation_rate=5, strides=1, padding='same', data_format="channels_first")(inputs)
        
        x = x_1 = Concatenate(axis=1)([x_1a, x_1b, x_1c])
        
        x = DropBlock2D(block_size=7, keep_prob=keep_prob, data_format="channels_first",sync_channels='true')(x)
        
        block_n=0
        for _ in range(n_id_block):
            block_n=block_n+1
            x = se_res_block_dilated_drop(x, n_id_block, block_n,filters=filters)
        
       # for _ in range(n_id_block2):
       #     x = se_res_block(x, filters=filters)     
            
        x = Conv2D(filters=30, kernel_size=3, strides=1, padding='same', data_format="channels_first")(x)

        x = Add()([x_1, x])

        x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', data_format="channels_first")(x)
        x = DropBlock2D(block_size=7, keep_prob=keep_prob, data_format="channels_first",sync_channels='true')(x)
        
        return Model(inputs=inputs, outputs=x)
    
    model = dadrsr()    
    
    adam= Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-08)

    model.compile(optimizer=adam, loss='mean_squared_error')

    # show the architecture and the parameters

    print(model.summary())

    # train the model
    print('reading training_dataset_ADT.nc')
    read_nc('training_dataset_ADT.nc')
    dim=tiles_input_training.shape
    jtile_dim=  dim[2]
    itile_dim = dim[3]
    nsample= dim[0]
    rand_order=np.random.permutation(nsample)
    tiles_input_training=tiles_input_training[rand_order,:,:,:]
    tiles_output_training=tiles_output_training[rand_order,:,:,:]

    print('start training')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    history = model.fit(tiles_input_training,tiles_output_training, batch_size=16, epochs=n_epochs, verbose=2, validation_split=val_split, callbacks=[es])
    
    model.save(model_name)
    print("Saved model to disk")

    train = history.history['loss']
    val = history.history['val_loss']
    # plot train and validation loss across multiple runs
    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='validation')
    plt.title('dilated ADR-SR ADT model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show(block=False)
    plt.savefig(model_dir+'dilated_drop_ADR-SR_CNN_ADT_loss.eps', dpi=150)
    plt.close()

else:

    print('training already completed')
