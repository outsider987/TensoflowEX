import tensorflow as tf
#让tensorflow的一些提示信息不出现，您可以去掉下面两行实验一下。立即会出现一个提示，说可以开启哪个开关，让程序效率更高一些。
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

w1 = tf.Variable(tf.random_normal([2,3],stddev=2))
w2 = tf.Variable(tf.random_normal([3,1],stddev=2))

x=tf.placeholder(tf.float32,shape=(1,2),name="input")
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

sess = tf.Session()
#在当前版本的tensorfloww中，initialize_all_variable()函数已经废弃，改用下面这个函数，看起来tensorflow的开发比较频繁，API也经常改变。希望它越变越好吧。
init=tf.global_variables_initializer()
sess.run(init)

print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))


product = tf.matmul([1,2],[1,2])
