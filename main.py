# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:27:48 2022

@author: Admin
"""


import argparse
from Hepler import Helper
from glob import glob
from Train import Train

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_path", help="path of trainig samples(G:/abstract_art_512/*)")
    parser.add_argument("image_height",help="height of an image",type=int)
    parser.add_argument("image_width",help="witdh of an image",type=int)
    parser.add_argument("image_channels",help="channels of an image",type=int)
    parser.add_argument("latent_dim",help="random point used to generate image",type=int)
    parser.add_argument("batch_size",help="training batch size",type=int)
    parser.add_argument("num_epochs",help="number of training epochs",type=int)
    parser.add_argument("output_sample",help="path for saving generated sample plot(abstract_art_512_samples)")
    
    return parser.parse_args()

if __name__== "__main__":

       
    args=arguments()
    training_path = args.training_path
    IMG_H = args.image_height
    IMG_W = args.image_width
    IMG_C = args.image_channels
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    num_epochs = args.num_epochs
    output_sample = args.output_sample
    
    training_path = glob(training_path)
    helper= Helper(training_path, batch_size, IMG_H, IMG_W)
    images_dataset = helper.tf_dataset(training_path, batch_size)
    image_shape=(IMG_H,IMG_W,IMG_C)
    
    T=Train(image_shape, latent_dim, num_epochs, images_dataset, batch_size, output_sample)
    T.train(image_shape, latent_dim, num_epochs, images_dataset, batch_size, output_sample)