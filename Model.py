# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:53:43 2022

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2DTranspose,LeakyReLU,Flatten,Dense,Dropout,BatchNormalization,Reshape,Activation,Conv2D
from tensorflow.keras.models import Model

class Model:
    
    def __init__(self,latent_dim,image_shape,**kwargs):
        self.latent_dim=latent_dim
        self.image_shape=image_shape
        
    def build_generator(self,latent_dim):
        """
        latent dimension for image creation
        """
        n_nodes = 8* 8*16
        noise = Input(shape=(latent_dim,), name="generator_noise_input")
        x = Dense(n_nodes, use_bias=False)(noise)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((8, 8, 16 ))(x)
        x=Conv2DTranspose(16, kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(32,kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(32,kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(32,kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(32,kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(3, kernel_size=3,strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        fake_output = Activation("tanh")(x)
        return Model(noise, fake_output, name="generator")
    
    
    def build_discriminator(self,image_shape):
        """
        image shape as tupe of (H,W,C)
        """
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        image_input = Input(shape=image_shape)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x=Conv2D(16, kernel_size=5, strides=2,kernel_initializer=w_init,padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Dense(8)(x)
        x = Dense(1)(x)
        return Model(image_input, x, name="discriminator")