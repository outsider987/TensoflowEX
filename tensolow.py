import tensorflow as tf

a = tf.constant('hello')
with tf.Session() as sess:
    print(sess.run(a))