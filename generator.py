
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, UpSampling2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import UpSampling3D, Conv2D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import scipy
import tensorflow as tf
from keras import backend as K
tf.set_random_seed(1)
import params as params
#import matplotlib.pyplot as plt

import numpy as np 

def generator_model(r, v, z):
        # latent, rep, seq

        # v = Reshape((-1,1,7))(v)
        # v = UpSampling2D(size=(16,16), data_format=None)(v) #broadcast 
        # r = Input(shape=r_shape)
        # z = Input(shape=z_shape)
        # v = Input(shape=v_shape)
        v1 = UpSampling2D(size=(28,28), data_format=None)(v)
        z1 = UpSampling2D(size=(28,28), data_format=None)(z)
        x = Concatenate(axis = 3)([r,v1,z1])
        
        x = Reshape((1,28,28,264))(x)
        s = x
        for i in range(params.num_frames-1):
            x = Concatenate(axis=1)([x,s])
        #x = Concatenate(axis=4)([x,r])
        
        # keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        # bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv1')(x)
        x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv1-1')(x)
        # print x.shape
        x = BatchNormalization(momentum=0.9)(x)
        x = UpSampling3D(size=(1,2,2))(x)
        
        x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv2')(x)
        x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv2-1')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = UpSampling3D(size=(1,2,2))(x)
        x = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv3')(x)
        x = Conv3D(128, (5, 5, 5), strides=(1, 1, 1), padding='same', activation='relu', name='conv3-1')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name='conv4-1')(x)
        output_clip = Conv3D(3, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='tanh', name='conv4')(x)
        # print output_clip.shape

        # Model([r,v,z], output_clip)

        return output_clip


if __name__ == '__main__':
    
    # v = Input(shape=params.v_shape)
    
    generator = generator_model()
    
    print(generator.summary())
    # train(epochs=20000, batch_size=32, sample_interval=200)
