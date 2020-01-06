import tensorflow as tf
import numpy as np
from config import Config, get_models_dir, get_predict_result_path


sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))


def jaccard_with_anchors(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    Introduction:
    @param anchors_min: shape=[totoal_anchors]
    @param anchors_max: shape=[totoal_anchors]
    @param len_anchors: the length of anchors ----> shape=[totoal_anchors]
    @param box_min: float, not array
    @param box_max: float, not array
    @return:
            jaccard: the iou between each anchors and a boxs
            for example:
                anchors_min = tf.constant([7, 15, 24])
    '''
    """Compute jaccard score between a box and the anchors.
    """

    int_xmin = tf.maximum(anchors_min, box_min)
    int_xmax = tf.minimum(anchors_max, box_max)

    inter_len = tf.maximum(int_xmax - int_xmin, 0.)

    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = tf.div(inter_len, union_len)
    return jaccard

def loop_condition(idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
                   b_match_x, b_match_w, b_match_labels, b_match_scores):
    '''
    A=[[1,2,3],
       [4,5,6]]
    i=[[1,2,3],
       [1,2,3]]
    r = tf.less(i, A)
    with tf.Session() as sess:
        print(sess.run(r))#[[False False False]
                          # [ True  True  True]]
    '''
    r = tf.less(idx, tf.shape(b_glabels))
    return r[0]


def loop_body(idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
              b_match_x, b_match_w, b_match_labels, b_match_scores):
    '''

    @param idx:
    @param b_anchors_rx: shape=[total_anchors]
    @param b_anchors_rw: shape=[total_anchors]
    @param b_glabels: shape=[None, num_classes]
    @param b_gbboxes: shape=[None, 3]
    @param b_match_x: shape=[total_anchors], init: 0
    @param b_match_w: shape=[total_anchors], init: 0
    @param b_match_labels: shape=[total_anchors, num_classes], init: [[1, 0, 0, ..., 0],
                                                                      [1, 0, 0, ..., 0],
                                                                                 .
                                                                                 .
                                                                                 .
                                                                      [1, 0, 0, ..., 0]]
    @param b_match_scores: shape=[total_anchors], init: 0
    @return:
    '''
    num_class = b_match_labels.get_shape().as_list()[-1]
    label = b_glabels[idx][0:num_class] # shape=[num_class]
    box_min = b_gbboxes[idx, 0] # float, not array
    box_max = b_gbboxes[idx, 1] # float, not array

    # ground truth
    box_x = (box_max + box_min) / 2
    box_w = (box_max - box_min)

    # predict
    anchors_min = b_anchors_rx - b_anchors_rw / 2 # [total_anchors]
    anchors_max = b_anchors_rx + b_anchors_rw / 2 # [total_anchors]

    len_anchors = anchors_max - anchors_min # [total_anchors]

    # jaccards: shape=[total_anchors]
    jaccards = jaccard_with_anchors(anchors_min, anchors_max, len_anchors, box_min, box_max)
    # jaccards > b_match_scores > -0.5 & jaccards > matching_threshold
    mask = tf.greater(jaccards, b_match_scores)
    matching_threshold = 0.5
    mask = tf.logical_and(mask, tf.greater(jaccards, matching_threshold))
    mask = tf.logical_and(mask, b_match_scores > -0.5)

    imask = tf.cast(mask, tf.int32)
    fmask = tf.cast(mask, tf.float32)
    # Update values using mask.
    # if overlap enough, update b_match_* with gt, otherwise not update
    b_match_x = fmask * box_x + (1 - fmask) * b_match_x
    b_match_w = fmask * box_w + (1 - fmask) * b_match_w

    ref_label = tf.zeros(tf.shape(b_match_labels), dtype=tf.int32)
    ref_label = ref_label + label
    b_match_labels = tf.matmul(tf.diag(imask), ref_label) + tf.matmul(tf.diag(1 - imask), b_match_labels)

    b_match_scores = tf.maximum(jaccards, b_match_scores)
    return [idx + 1, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
            b_match_x, b_match_w, b_match_labels, b_match_scores]





def anchor_bboxes_encode(anchors, glabels, gbboxes, Index, config, layer_name, pre_rx=None, pre_rw=None):
    '''
    :param anchors: output of the network ----> shape=[bs, totoal anchors, classes + 3]  3: overlap, d_c, d_w
    :param glabels: ground truth classes ----> shape=[None, num_classes]
    :param gbboxes:  ground truth bounding boxes ----> shape=[None, 3] 3: min, max,
    :param Index: shape=[bsz + 1]
    :param config:
    :param layer_name: input param of variable scope
    :return:
            batch_match_x: corresponding to anchors_rx
            batch_match_w: corresponding to anchors_rw
            batch_match_labels: corresponding to anchors_class
            batch_match_scores: corresponding to anchors_conf
            anchors_class: outpyt of the anchor layer ----> shape=[bs, totoal_anchors, num_class]
            anchors_conf: outpyt of the anchor layer: overlap ----> shape=[bs, totoal_anchors]
            anchors_rx: outpyt of the anchor layer: center, not offset center ----> shape=[bs, totoal_anchors]
            anchors_rw: outpyt of the anchor layer: width, not offset width ----> shape=[bs, totoal_anchors]
    '''
    batch_match_x = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_w = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_scores = tf.reshape(tf.constant([]), [-1, 4])
    batch_match_labels = tf.reshape(tf.constant([], dtype=tf.int32),
                                    [-1, 4, 4])
    dtype = tf.float32

    anchors_class = anchors[:, :, 0:4]
    anchors_conf = anchors[:, :, -1]
    anchors_x = anchors[:, :, 4]
    anchors_w = anchors[:, :, 5]

    for i in range(3):
        shape = (4)
        match_x = tf.zeros(shape, dtype)
        match_w = tf.zeros(shape, dtype)
        match_scores = tf.zeros(shape, dtype)

        match_labels_other = tf.ones((4, 1), dtype=tf.int32)
        match_labels_class = tf.zeros((4, 4 - 1), dtype=tf.int32)
        match_labels = tf.concat([match_labels_other, match_labels_class], axis=-1) # shape=[total_anchors, num_classes]

        b_anchors_rx = anchors_x[i] # shape=[num_anchors * num_dbox]
        b_anchors_rw = anchors_w[i] # shape=[num_anchors * num_dbox]

        b_glabels = glabels[Index[i]:Index[i + 1]] # shape=[None, num_classes]
        b_gbboxes = gbboxes[Index[i]:Index[i + 1]] # shape=[None, 3]

        idx = 0
        [idx, b_anchors_rx, b_anchors_rw, b_glabels, b_gbboxes,
         match_x, match_w, match_labels, match_scores] = \
            tf.while_loop(loop_condition, loop_body,
                          [idx, b_anchors_rx, b_anchors_rw,
                           b_glabels, b_gbboxes,
                           match_x, match_w, match_labels, match_scores])

        match_x = tf.reshape(match_x, [-1, 4])
        batch_match_x = tf.concat([batch_match_x, match_x], axis=0)

        match_w = tf.reshape(match_w, [-1, 4])
        batch_match_w = tf.concat([batch_match_w, match_w], axis=0)

        match_scores = tf.reshape(match_scores, [-1, 4])
        batch_match_scores = tf.concat([batch_match_scores, match_scores], axis=0)

        match_labels = tf.reshape(match_labels, [-1, 4, 4])
        batch_match_labels = tf.concat([batch_match_labels, match_labels], axis=0)

    return [batch_match_x, batch_match_w, batch_match_labels, batch_match_scores,
            anchors_class, anchors_conf, anchors_x, anchors_w]

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
config = Config()
batch_match_x, batch_match_w, batch_match_labels, batch_match_scores, \
anchors_class, anchors_conf, anchors_rx, anchors_rw = anchor_bboxes_encode(anchors, glabels, gbboxes, Index, config, 'AL1', pre_rx=None, pre_rw=None)

# with tf.Session() as sess:
batch_match_x, batch_match_w, batch_match_labels, batch_match_scores, \
anchors_class, anchors_conf, anchors_rx, anchors_rw = sess.run([batch_match_x, batch_match_w, batch_match_labels, batch_match_scores,
                                                                anchors_class, anchors_conf, anchors_rx, anchors_rw])
print('batch_match_x\n', batch_match_x)
print('batch_match_w\n', batch_match_w)
print('batch_match_labels\n', batch_match_labels)
print('batch_match_scores\n', batch_match_scores)
print('anchors_class\n', anchors_class)
print('anchors_conf\n', anchors_conf)
print('anchors_rx\n', anchors_rx)
print('anchors_rw\n', anchors_rw)
