import tensorflow as tf
import numpy as np

# feat = tf.zeros(shape=[10, 2], dtype=tf.float32)

start_end = tf.constant([[0.16210938, 0.30078125],
                         [0.59570312, 0.81640625],
                         [0.58203125, 0.75976562],
                         [0.59960938, 0.85351562]])


# center = tf.reshape(tf.constant([[0.1, 0.3, 0.2, 0.44, 0.5]]), [-1, 1])
#
# width = tf.reshape(tf.constant([[14, 12, 23, 14, 65]]), [-1, 1])

def mx_width_label(num_neure, start_end):
    """
    Introduction:
    @param num_neure:
    @param start_end: shape=[-1, 2]
    @return:
    """
    feat = tf.zeros(shape=[num_neure, 2], dtype=tf.float32)
    center = (start_end[:, 1:2] + start_end[:, 0:1]) / 2.0
    width = start_end[:, 1:2] - start_end[:, 0:1]

    num_nerue = tf.shape(feat)[0]
    interval = 1 / tf.cast(num_nerue, tf.float32)
    norm_center = center / interval
    norm_width = width / interval


    def mx_get_label(feat, norm_center, norm_width):
        idx_grid = norm_center.astype(np.int32)
        norm_center = norm_center - idx_grid
        idx_grid = np.reshape(idx_grid, [-1])
        feat[idx_grid] = np.concatenate((norm_center, norm_width), axis=-1)
        return feat

    [Y] = tf.py_func(mx_get_label, [feat, norm_center, norm_width], [tf.float32])
    return norm_center, norm_width, Y

def decode_box(output):
    """

    @param output: shape=[-1,2]
    @return:
    """
    num_nerue = tf.shape(output)[0]
    idx_grid = tf.cast(tf.range(num_nerue) , tf.float32)
    idx_grid = tf.reshape(idx_grid, [-1,1])
    interval = 1 / tf.cast(num_nerue, tf.float32)

    norm_center = output[:, 0:1]
    norm_width = output[:, 1:2]

    center = interval * (idx_grid + norm_center)
    width = norm_width * interval

    start = center - width / 2.0
    end = center + width / 2.0

    start_end = tf.concat((start, end), axis=-1)
    start_end_mask = tf.zeros_like(start_end)

    start_end = tf.where(output > 0, start_end, start_end_mask)

    return start_end



norm_center, norm_width, Y = mx_width_label(10, start_end)
valid_idx = decode_box(Y)


with tf.Session() as sess:
    norm_center, norm_width, Y = sess.run([norm_center, norm_width, Y])
    [valid_idx] = sess.run([valid_idx])
    print('norm_center: \n',norm_center)
    print('norm_width: \n',norm_width)
    print('Y: \n',Y)

    print('valid_idx:', valid_idx)