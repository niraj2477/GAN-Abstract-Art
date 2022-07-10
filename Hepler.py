# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:16:33 2022

@author: Admin
"""
import tensorflow as tf
from sklearn.utils import shuffle
class Helper:
    def __init__(self,training_path,batch_size,image_height,image_width):
        self.training_path=training_path
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
    
    def load_image(self,image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize_with_crop_or_pad(img, self.image_height,self.image_width)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        return img
    
    def tf_dataset(self,training_path,batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(training_path)
        dataset = dataset.shuffle(buffer_size=10240)
        dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset