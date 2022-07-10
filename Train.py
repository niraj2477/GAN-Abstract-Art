# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:03:37 2022

@author: Admin
"""

from Gan import Gan
from Model import Model
import tensorflow as tf
from matplotlib import pyplot
import numpy as np
class Train:
    
    def __init__(self,image_shape,latent_dim,num_epochs,images_dataset,batch_size,output_sample):
        self.image_shape= image_shape
        self.latent_dim= latent_dim
        self.num_epochs= num_epochs
        self.images_dataset= images_dataset
        self.batch_size= batch_size
        self.output_sample= output_sample
        
    def save_plot(examples, epoch, n,output_sample):
        examples = (examples + 1) / 2.0
        for i in range(n * n):
            pyplot.subplot(n, n, i+1)
            pyplot.axis("off")
            pyplot.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
        filename = f"{output_sample}/generated_plot_epoch-{epoch+1}.png"
        pyplot.savefig(filename)
        pyplot.close()
    
    def train(self,image_shape,latent_dim,num_epochs,images_dataset,batch_size,output_sample):
        
        """
        image shape as tupe of (H,W,C)
        latent dimension for image creation
        """
        d_model= Model.build_discriminator(self, image_shape)
        g_model= Model.build_generator(self, latent_dim)
        gan = Gan(d_model, g_model, latent_dim)
    
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.01)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        gan.compile(d_optimizer, g_optimizer, bce_loss_fn)
        
        
        for epoch in range(num_epochs):
           gan.fit(images_dataset, epochs=1)
           g_model.save("saved_models/art_g_model.h5")
           d_model.save("saved_models/art_d_model.h5")
    
           n_samples = 16
           noise = np.random.normal(size=(n_samples, latent_dim))
           examples = g_model.predict(noise)
           self.save_plot(examples, epoch, int(np.sqrt(n_samples)),output_sample)
        