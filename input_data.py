import numpy as np 
import tensorflow as tf 
import os
import importlib as plt
import matplotlib as plt




def get_file(filepath):
    # the each image
    images = []
    # the each folder
    subfolders = []
    
    for dirpath,dirname,filename in os.walk(filepath):
        for name in filename:
            images.append(os.path.join(dirpath,name))

        for name in dirname:
            subfolders.append(os.path.join(dirpath,name))    
    labels = []
    counts = 0
    for a_folder in subfolders:
        n_imgs = len(os.listdir(a_folder))
        labels = np.append(labels,n_imgs*[counts])
        counts +=1   

    subfolders = np.array([images,labels])
    subfolders = subfolders.transpose()

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i))for i in label_list]

    return image_list, label_list


def get_batchsize(imagelist, label_list,IMG_WIDTH,IMG_HEIGHT,BATCH_SIZE,CAPACITY):
    imagelist = tf.cast(imagelist,tf.string)
    label_list = tf.cast(label_list,tf.int32)

    input_queue = tf.train.slice_input_producer([imagelist,label_list])

    label_list = input_queue[1]
    image_content = tf.read_file(input_queue[0])
    imagelist = tf.image.decode_jpeg(image_content,channels=3)

    imagelist = tf.image.resize_image_with_crop_or_pad(imagelist,IMG_WIDTH,IMG_HEIGHT)
    imagelist = tf.image.per_image_standardization(imagelist)
    image_batch,label_batch = tf.train.batch([imagelist,label_list],batch_size=BATCH_SIZE,num_threads=64,capacity=CAPACITY)

    label_batch = tf.reshape(label_batch,[BATCH_SIZE])
    return image_batch,label_batch




# def main():
#     IMG_HEIGHT = 208
#     IMG_WIDTH = 208
#     BATCH_SIZE =2 
#     CAPACITY = 256

#     datapath =os.getcwd()
#     datapath +="\\face_data"
#     imagelist,labelist = get_file(datapath)
#     image_batch,label_batch =  get_batchsize(imagelist,labelist,IMG_WIDTH,IMG_HEIGHT,BATCH_SIZE,CAPACITY)
#     with tf.Session() as sess:
#         i = 0
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         try:
#             while not coord.should_stop() and i<1:
#                 img, label = sess.run([image_batch,label_batch])

#                 for j in np.arange(BATCH_SIZE):
#                     print("label:%d"%label[j])
#                     plt.imshow(img[j,:,:,:])
#                     plt.show()
#                 i+=1
#         except tf.errors.OutOfRangeError:
#             print("done!")
#         finally:
#             coord.request_stop()
#         coord.join(threads)

# if __name__ == "__main__":
#     main()


