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



# filename_queue = tf.train.string_input_producer()



def main():

    cwd = os.getcwd()
    TargetPath = cwd+"\\TensoflowEX\\face_data"
    fileNames=[]
    fileNames = GetfilenameFunction(TargetPath)
    #產生文件名對列
    filename_queue = tf.train.string_input_producer(fileNames,shuffle=False, num_epochs=1)
if __name__ == '__main__':
    main()
