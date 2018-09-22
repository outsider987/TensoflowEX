import tensorflow  as tf
import os
import numpy as np 

def GetfilenameFunction(TargetPath):
 for filename,dirname,dirpath in os.walk(TargetPath)
    for dirpath


filename_queue = tf.train.string_input_producer()



def main():

    cwd = os.getcwd()
    TargetPath = cwd+"\\TensoflowEX\\face_data"
    GetfilenameFunction(TargetPath)
if __name__ == '__main__':
    main()
