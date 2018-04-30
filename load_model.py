from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import scipy.misc, numpy as np, os, sys, transform


def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img


file_name = '../rain_princess.ckpt'
img_path = 'images/chicago.jpg'
FLAGS = None

batch_shape = (1, 256, 256, 3)
img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

preds = transform.net(img_placeholder)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, file_name)

    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
        print(i)
        
        array = i.eval(sess)
        print(i.shape)
        arr = array.flatten()
        # print(arr)
        np.savetxt('data/'+i.name+'.out', arr, delimiter='\n')
