import tensorflow  as tf
import os
import numpy as np 

def GetfilenameFunction(TargetPath):
    images = []
    subfolders = []
    for dirPath, dirNames, fileNames in os.walk(TargetPath):
        for name in fileNames:
           images.append(os.path.join(dirPath, name))
        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

    return fileNames

def read_and_decode(fileNames, batch_size): 
    #產生文件名對列
    filename_queue = tf.train.string_input_producer(fileNames,shuffle=False, num_epochs=1)
    
    #數據讀取器
    reader = tf.TFRecordReader()
    key,serialized_example =reader.read(filename_queue)
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                       'gril_raw': tf.FixedLenFeature([], tf.string), })

    image = tf.decode_raw(img_features['gril_raw'], tf.uint8)
    image = tf.reshape(image, [42, 42])
    
    label = tf.cast(img_features['Label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size=batch_size,
                                 capacity=10000 + 3 * batch_size,
                                 min_after_dequeue=1000)

    return image_batch, label_batch



def main():

    #Tf parmater instiall
    Batch_Size  = 1
    

    cwd = os.getcwd()
    TargetPath = cwd+"\\TensoflowEX\\face_data"
    fileNames=[]
    fileNames = GetfilenameFunction(TargetPath)
    read_and_decode(fileNames,Batch_Size)
if __name__ == '__main__':
    main()
