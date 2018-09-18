import tensorflow as tf 
import os
import cv2
import numpy as np
 

# TFwriter = tf.python_io.TFRecordWriter("\\faceTF.tfrecords")
def get_file(file_dir):
    images= []
    subfolders = []

    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))
        
        
        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))


    labels = []
    count = 0
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count+=1

    subfolders = np.array([images, labels])
    subfolders = subfolders.transpose()

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list
        

def main ():
    
    cwd = os.getcwd()
    root = cwd+"\\TensoflowEX\\face_data"
    file_dir = cwd

    
    # 取回所有檔案路徑
    images, labels = get_file(root)
    test = images
    

if __name__ == '__main__':
    main()
        
     

