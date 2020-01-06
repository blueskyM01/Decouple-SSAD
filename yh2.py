import tensorflow as tf
import numpy as np



def mx_IOU(anchors, gt_boxes):
    '''
    Introduction: 计算iou

    输入为：
    anchors: [[0,10],                        gt_boxes:[[0,10],
             [2,7],                                    [3,6],
             [3,4],                                    [9,10]]
             [6,8]]

    输出为：
    iou:
            [[1.         0.3        0.1       ]
             [0.5        0.6        0.        ]
             [0.1        0.33333333 0.        ]
             [0.2        0.         0.        ]]

    从上面可以看出： 是将“anchors”中的“每行”依次与“gt_boxes”中的“所有行”计算iou，
                  最终的输出iou：其“每行”对应着“anchors”中的“每行”与“gt_boxes”中的“所有行”计算出的iou
    @param anchors: shape: [-1,2] 含义:每行(start, end)
    @param gt_boxes: shape: [-1,2] 含义:每行(start, end)
    @return:
    '''

    def mx_IOU_py_function(anchors, gt_boxes):
        anchors = np.expand_dims(anchors, axis=1)
        u0 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 0]
        u1 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 1]
        i0 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 0]
        i1 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 1]
        iou = (i1 - i0) / (u1 - u0)
        iou = np.maximum(iou, 0)
        iou = iou.astype(np.float32)
        return iou

    iou = tf.py_func(mx_IOU_py_function, [anchors, gt_boxes], [tf.float32])

    return iou[0]

def mx_MatchAnchors(anchors, glabels, gbboxes, Index):


    batch_match_x = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_w = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_scores = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_labels = tf.reshape(tf.constant([], dtype=tf.int32), [-1, 4, 4])

    batch_size = 3
    dtype = tf.float32
    shape = (4)

    anchors_class = anchors[:, :, 0:4]
    anchors_conf = anchors[:, :, -1]

    anchors_start_end = anchors[:, :, 4:6]
    gbboxes = gbboxes[:, 0:2]

    for i in range(batch_size):

        match_x = tf.zeros(shape, dtype)
        match_w = tf.zeros(shape, dtype)
        # match_scores = tf.zeros(shape, dtype)

        match_labels_other = tf.ones((4, 1), dtype=tf.int32)
        match_labels_class = tf.zeros((4, 4 - 1), dtype=tf.int32)
        mask_match_labels = tf.concat([match_labels_other, match_labels_class], axis=-1)  # shape=[total_anchors, num_classes]

        center_width = anchors_start_end[i]

        start = center_width[:, 0:1] - center_width[:, 1:2] / 2
        end = center_width[:, 0:1] + center_width[:, 1:2] / 2
        start_end = tf.concat((start, end), -1)

        b_glabels = glabels[Index[i]:Index[i + 1]]  # shape=[None, num_classes]
        b_gbboxes = gbboxes[Index[i]:Index[i + 1]]  # shape=[None, 3]

        iou = mx_IOU(start_end, b_gbboxes)
        best_idx = tf.cast(tf.arg_max(iou, dimension=1), tf.int32)
        num_anchors = start_end.get_shape().as_list()[0]

        mask_iou = tf.zeros_like(iou)
        best_class_iou = tf.where(iou > 0.5, iou, mask_iou)
        best_class_idx = tf.reduce_sum(best_class_iou, axis=1)
        # ----------------------batch_match_labels----------------------------------------------------------------
        match_labels_mask = tf.gather(b_glabels, best_idx, axis=0)
        match_labels = tf.where(best_class_idx > 0, match_labels_mask, mask_match_labels)
        batch_match_labels = tf.concat((batch_match_labels, tf.reshape(match_labels, [-1, 4, 4])), axis=0)

        # ----------------------batch_match_x----------------------------------------------------------------
        match_x_mask = tf.gather((b_gbboxes[:, 0] + b_gbboxes[:, 1]) / 2, best_idx, axis=0)
        match_x_zeros = tf.zeros_like(match_x_mask)
        match_x = tf.where(best_class_idx > 0, match_x_mask, match_x_zeros)
        batch_match_x = tf.concat((batch_match_x, tf.reshape(match_x, [-1, 4])), axis=0)

        # ----------------------batch_match_w----------------------------------------------------------------
        match_w_mask = tf.gather((b_gbboxes[:, 1] - b_gbboxes[:, 0]), best_idx, axis=0)
        match_w_zeros = tf.zeros_like(match_w_mask)
        match_w = tf.where(best_class_idx > 0, match_w_mask, match_w_zeros)
        batch_match_w = tf.concat((batch_match_w, tf.reshape(match_w, [-1, 4])), axis=0)

        # ----------------------batch_match_scores----------------------------------------------------------------
        best_overlap_idx = tf.reshape(best_idx, [-1, 1])
        rows_idx = tf.expand_dims(tf.range(num_anchors), -1)
        best_iou_idx = tf.concat((rows_idx, best_overlap_idx), axis=-1)
        match_scores = tf.gather_nd(iou, best_iou_idx)
        batch_match_scores = tf.concat((batch_match_scores, tf.reshape(match_scores, [-1,4])), axis=0)




    return batch_match_scores, batch_match_labels, batch_match_x, batch_match_w

anchors = tf.constant([[[0, 0, 1, 0, 0.23, 0.07, 0.52],
                        [0, 1, 0, 0, 0.2, 0.8, 0.40],
                        [0, 0, 1, 0, 0.7, 0.21, 0.72],
                        [0, 0, 0, 1, 0.15, 0.37, 0.14]],
                       [[0, 0, 1, 0, 0.55, 0.27, 0.32],
                        [0, 1, 0, 0, 0.29, 0.18, 0.01],
                        [0, 0, 1, 0, 0.60, 0.208, 0.93],
                        [0, 0, 0, 1, 0.67, 0.17, 0.64]],
                       [[0, 0, 1, 0, 0.23, 0.07, 0.32],
                        [0, 1, 0, 0, 0.69, 0.18, 0.01],
                        [0, 0, 1, 0, 0.37, 0.208, 0.93],
                        [0, 0, 0, 1, 0.67, 0.17, 0.64]]
                       ])
glabels = tf.constant([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
gbboxes = tf.constant([[0.16210938, 0.30078125, 1.],
                       [0.59570312, 0.81640625, 1.],
                       [0.58203125, 0.75976562, 1.],
                       [0.59960938, 0.85351562, 1.]])
Index = tf.constant([0, 2, 3, 4])

batch_match_scores, batch_match_labels, batch_match_x, batch_match_w = mx_MatchAnchors(anchors, glabels, gbboxes, Index)
with tf.Session() as sess:
    batch_match_scores_, batch_match_labels_, \
    batch_match_x_, batch_match_w_ = sess.run([batch_match_scores, batch_match_labels, batch_match_x, batch_match_w])
    print('batch_match_scores_:\n', batch_match_scores_)
    print('batch_match_labels_:\n', batch_match_labels_)
    print('batch_match_x_:\n', batch_match_x_)
    print('batch_match_w_:\n', batch_match_w_)