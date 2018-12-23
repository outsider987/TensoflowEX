import tensorflow as tf 
import numpy as np 
import matplotlib as plt


def inference(images,batch_size,n_classes):
    
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
        shape=[3,3,3,16],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable("biases",
        shape=[16],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 =tf.nn.relu(pre_activation,name = scope.name)
