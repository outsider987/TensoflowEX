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


def read_and_decode(filename,bathsize):
    filename_queue  = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    
   # 數據解析
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                       'image_raw': tf.FixedLenFeature([], tf.string), })
    
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [42, 42])
    
    label = tf.cast(img_features['Label'], tf.int64)
    batch_size = 1000
    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size=batch_size,
                                 capacity=10000 + 3 * batch_size,
                                 min_after_dequeue=1000)

    return image_batch, label_batch



def main():
    global image_batch,label_batch
    Path = os.getcwd()
    Path += "\\Train.tfrecords"
    image_batch, label_batch =read_and_decode(Path,20)
    



   
    

if __name__ == '__main__':
    main()

global image_batch,label_batch,image_batch_train,label_batch_train
# 轉換陣列的形狀
image_batch_train = tf.reshape(image_batch, [-1, 42*42])
# 把 Label 轉換成獨熱編碼
Label_size =10
label_batch_train = tf.one_hot(label_batch, Label_size)
# W 和 b 就是我們要訓練的對象
W = tf.Variable(tf.zeros([42*42, Label_size]))
b = tf.Variable(tf.zeros([Label_size]))

#   我們的影像資料，會透過 x 變數來輸入   
x = tf.placeholder(tf.float32, [None, 42*42])
# 這是參數預測的結果
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 這是每張影像的正確標籤
y_ = tf.placeholder(tf.float32, [None, 10])
# 計算最小交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

# 使用梯度下降法來找最佳解
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# 計算預測正確率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化是必要的動作
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    sess.run(init_op) 
    
    # 建立執行緒協調器
    coord = tf.train.Coordinator()
    
    # 啟動文件隊列，開始讀取文件
    threads = tf.train.start_queue_runners(coord=coord)
    
    # 迭代 10000 次，看看訓練的成果
    for count in range(10000):     
        # 這邊開始讀取資料
        image_data, label_data = sess.run([image_batch_train, label_batch_train])
   
        # 送資料進去訓練
        sess.run(train_step, feed_dict={x: image_data, y_: label_data})
        
        # 這裡是結果展示區，每 10 次迭代後，把最新的正確率顯示出來
        if count % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
            print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy*100))

    # 結束後記得把文件名隊列關掉
    coord.request_stop() 
    coord.join(threads)

    # Path += "\\TensoflowEX\\face_data"
    # getfile(Path) 
