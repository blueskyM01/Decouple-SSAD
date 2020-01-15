import numpy as np
import numpy
import os, time
import tensorflow as tf
from os.path import join
import mx_utils

# def mx_fuse_anchor_box(prop_width, fuse_threshold=0.5):
#     """
#     :param prop_width: shape=[bs, num_neure, 1]
#     :param fuse_threshold:
#     """
#     with_set = tf.reshape(prop_width, shape=[-1, 1])
#     num_neure = prop_width.get_shape().as_list()[1]
#     center_set = tf.cast(tf.range(num_neure), tf.float32) + tf.constant([0.5])
#     center_set = tf.reshape(center_set, shape=[-1,1])
#
#     start = center_set - with_set / 2
#     end = center_set + with_set / 2
#     start_end = tf.concat((start, end), axis=-1)
#     start_end = tf.clip_by_value(start_end, 0, num_neure)
#
#     def mx_FuseAnchor(start_end, fuse_threshold):
#         """
#
#         :param start_end:
#         :param fuse_threshold:
#         """
#         anchor = start_end.tolist()
#         fuse_idx = []
#         for idx in range(1, len(anchor) - 1):
#             span_t_last = anchor[idx - 1]
#             span_t = anchor[idx]
#             span_t_next = anchor[idx + 1]
#             iou1 = mx_utils.temporal_iou(span_t_last, span_t)
#             iou2 = mx_utils.temporal_iou(span_t_next, span_t)
#
#             if max(iou1, iou2) > fuse_threshold:
#                 if iou1 < iou2:
#                     start = min(span_t_next[0], span_t[0])
#                     end = max(span_t_next[1], span_t[1])
#                     fuse_idx.append([idx, idx + 1])
#                 else:
#                     start = min(span_t_last[0], span_t[0])
#                     end = max(span_t_last[1], span_t[1])
#                     fuse_idx.append([idx - 1, idx])
#
#         # 去除重复的
#         fuse_idx_clean = []
#         for r in fuse_idx:
#             if not r in fuse_idx_clean:
#                 fuse_idx_clean.append(r)
#
#         # get fuse proposal
#         fuse_proposal_clean = []
#         for f_i in fuse_idx_clean:
#             span_t_last = anchor[f_i[0]]
#             span_t = anchor[f_i[1]]
#             start = min(span_t_last[0], span_t[0])
#             end = max(span_t_last[1], span_t[1])
#             fuse_proposal_clean.append([start, end])
#
#         final_fuse = mx_utils.mx_GetFuseAndUnfuseIdx(fuse_idx_clean, anchor)
#         return final_fuse
#
#     [final_fuse] = tf.py_func(mx_FuseAnchor, [start_end, fuse_threshold], [tf.int32])
#
#     return final_fuse
#
#
# prop_width = tf.reshape(tf.constant([2.4, 2.5, 1, 2.4]), shape=[1, -1, 1])
# fuse_threshold = 0.5
#
# final_fuse = mx_fuse_anchor_box(prop_width, fuse_threshold)
#
# with tf.Session() as sess:
#     a = sess.run(final_fuse)
#     print(a)

a = np.array([[1,2],
              [3,4],
              [5,6]])

b = np.array([0,2])
c = np.array([1,1])
print(a[b, c])