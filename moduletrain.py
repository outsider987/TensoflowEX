import tensorflow  as tf
import os
import numpy as np 



def getfile(Path):
     # the each image
    images = []
    # the each folder
    subfolders = []
    
    for dirpath,dirname,filename in os.walk(Path):
        for name in filename:
            images.append(os.path.join(dirpath,name))

        for name in dirname:
            subfolders.append(os.path.join(dirpath,name))
        
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
        

def main():
    Path = os.getcwd()
    Path += "\\TensoflowEX\\face_data"
    getfile(Path)
    

if __name__ == '__main__':
    main()