import tensorflow as tf 
import numpy as np 
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
    datapath +="\\face_data"
    logits_path =os.getcwd()
    logits_path +="\\logits"
    imagelist,labelist =input_data.get_file(datapath)
    image_batch,label_batch = input_data.get_batchsize(imagelist,labelist,IMG_WIDTH,IMG_HEIGHT,BATCH_SIZE,CAPACITY)

    train_logits =model.inference(image_batch,BATCH_SIZE ,N_CLASSES )
    train_loss = model.losses(train_logits,label_batch)
    train_op = model.trainning(train_loss,learning_rate)
    train_acc = model.evaluation(train_logits,label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logits_path,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord =tf.train.Coordinator()
    Threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop() :
                break
            _,tra_loss,tra_acc =sess.run([train_op,train_loss,train_acc]) 

            if step % 50 == 0:
                print("Step %d, train_loss = %.2f,train accuarcy = %.2f%%" %(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logits_path,"model.ckpt")
                saver.save(sess,checkpoint_path,global_step=step)

    except tf.errors.OutOfRangeError:
        print("Done Trainng -- epoch limit reached")
    finally:
        coord.request_stop()
    

    coord.join(Threads)
    sess.close()

from PIL import Image
import matplotlib.pyplot as plt
def get_one_image (train):
    n=len(train)
    ind = np.random.randint(0,n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([IMG_HEIGHT,IMG_WIDTH])
    image = np.array(image)
    return image
    

def evelaut_one_image():
    train_dir = os.getcwd()
    train_dir += "\\face_data"
    train, train_label  = input_data.get_file(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image =tf.cast(image_array,tf.float32)
        image =tf.reshape(image,[1,IMG_WIDTH,IMG_HEIGHT,3])
        logit = model.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,shape=[IMG_WIDTH,IMG_HEIGHT,3])

        logs_train_dir = os.getcwd()
        logs_train_dir +="\\logits"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("Loading Sucess,golbal_step is %s" % global_step)
            else:
                print("no checkpoint file found")

            prediction = sess.run(logit,feed_dict = {x:image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print("this is a cat with possibility %.6f" %prediction[:,0])
            else:
                print("this is a dog with possibility %.6f" %prediction[:,1])


if __name__ == "__main__":
    evelaut_one_image()