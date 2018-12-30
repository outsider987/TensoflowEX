import tensorflow as tf 
import numpy as np 
import matplotlib as plt
import input_data
import model
import os

N_CLASSES = 2
IMG_HEIGHT = 208
IMG_WIDTH = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000
learning_rate = 0.0001


def run_training():
    
    datapath =os.getcwd()
    datapath +="\\TensoflowEX\\face_data"
    imagelist,labelist =input_data.get_file(datapath)
    image_batch,label_batch = input_data.get_batchsize(imagelist,labelist,IMG_WIDTH,IMG_HEIGHT,BATCH_SIZE,CAPACITY)

    train_logits =model.inference(image_batch,BATCH_SIZE ,N_CLASSES )
    train_loss = model.losses(train_logits,label_batch)
    train_op = model.trainning(train_loss,learning_rate)
    train_acc = model.evaluation(train_logits,label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(datapath,sess.graph)
    saver = tf.train.start_queue_runners(sess=sess)