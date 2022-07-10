# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:55:38 2022

@author: Admin
"""
import numpy as np

from tensorflow.keras.models import load_model
from matplotlib import pyplot

if __name__== "__main__":
    
    model = load_model("saved_models/art_g_model.h5",compile=False)
    n_samples = 1   ## n should always be a square of an integer.
    latent_dim = 200
    latent_points = np.random.normal(size=(n_samples, latent_dim))
    examples = model.predict(latent_points)
    examples = (examples + 1) / 2.0
    pyplot.imshow(examples[0])