import tensorflow as tf
import numpy as np

anchors = tf.constant([[0,10],
                       [2,7],
                       [3,4],
                       [6,8]])
mask = tf.zeros_like(anchors)
c = tf.constant([2,0,0,4])

d = tf.where(c > 0, anchors, mask)

with tf.Session() as sess:

    e = sess.run(d)
    print(e)