# -*- coding: utf-8 -*-
"""
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------

Train, test and post-processing for the main stream of Decouple-SSAD
Improved version of SSAD

Usage:
Please refer to `run.sh` for details.
e.g.
`python main_stream.py test UCF101 temporal main_stream main_stream`

"""

from operations import *
from load_data import get_train_data, get_test_data
from config import Config, get_models_dir, get_predict_result_path
import time
from os.path import join
import sys
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
####################################### PARAMETERS ########################################


parser = argparse.ArgumentParser()

# -----------------------------m4_BE_GAN_network-----------------------------
parser.add_argument("--stage", default='train_test_fuse', type=str, help="train/test/fuse/train_test_fuse")
parser.add_argument("--pretrain_dataset", default='UCF101', type=str, help="UCF101/KnetV3")
parser.add_argument("--mode", default='temporal', type=str, help="temporal/spatial")
parser.add_argument("--method", default='main_stream', type=str, help="main_stream")
parser.add_argument("--method_temporal", default='main_stream', type=str, help="main_stream")

parser.add_argument("--log_dir", default='./logs/', type=str, help="temporal/spatial")
parser.add_argument("--model_dir", default='./models/', type=str, help="main_stream")
parser.add_argument("--results_dir", default='./results/', type=str, help="main_stream")

cfg = parser.parse_args()
# stage = sys.argv[1]  # train/test/fuse/train_test_fuse
# pretrain_dataset = sys.argv[2]  # UCF101/KnetV3
# mode = sys.argv[3]  # temporal/spatial
# method = sys.argv[4]
# method_temporal = sys.argv[5]  # used for final result fusing

# stage = 'train_test_fuse'  # train/test/fuse/train_test_fuse
# pretrain_dataset = 'UCF101'  # UCF101/KnetV3
# mode = 'temporal'  # temporal/spatial
# method = 'main_stream'
# method_temporal = 'main_stream'  # used for final result fusing

global_step = tf.Variable(0, name='global_step', trainable=False)

if (cfg.mode == 'spatial' and cfg.pretrain_dataset == 'Anet') or cfg.pretrain_dataset == 'KnetV3':
    feature_dim = 2048
else:
    feature_dim = 1024

models_dir = get_models_dir(cfg.model_dir, cfg.mode, cfg.pretrain_dataset, cfg.method)
models_file_prefix = join(models_dir, 'model-ep')
test_checkpoint_file = join(models_dir, 'model-ep-30')
predict_file = get_predict_result_path(cfg.results_dir, cfg.mode, cfg.pretrain_dataset, cfg.method)


######################################### TRAIN ##########################################

def train_operation(X, Y_label, Y_bbox, Index, LR, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    # MALs = main_anchor_layer(net)
    MALs, N_lsit, width_list = mx_fuse_anchor_layer(net, config)

    # --------------------------- Main Stream -----------------------------
    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_width = tf.reshape(tf.constant([]), [-1, 2])

    full_mainAnc_BM_x = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_w = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_labels = tf.reshape(tf.constant([], dtype=tf.int32), [bsz, -1, ncls])
    full_mainAnc_BM_scores = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_width = tf.reshape(tf.constant([]), [-1, 2])


    for i, ln in enumerate(config.layers_name):
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')

        # --------------------------- Fuse Loss -----------------------------
        width = width_list[i]
        width = tf.reshape(width, [-1, 2])
        num_neure = tf.shape(width)[0]
        start_end = Y_bbox[:, 0:2]
        Y_Fuse_label = mx_width_label(num_neure, start_end)

        # --------------------------- Main Stream -----------------------------
        [mainAnc_BM_x, mainAnc_BM_w, mainAnc_BM_labels, mainAnc_BM_scores,
         mainAnc_class, mainAnc_conf, mainAnc_rx, mainAnc_rw] = \
            anchor_bboxes_encode(mainAnc, Y_label, Y_bbox, Index, config, ln, N_lsit[i])

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat([full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat([full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat([full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat([full_mainAnc_xmax, mainAnc_xmax], axis=1)
        full_mainAnc_width = tf.concat([full_mainAnc_width, width], axis=0)

        full_mainAnc_BM_x = tf.concat([full_mainAnc_BM_x, mainAnc_BM_x], axis=1)
        full_mainAnc_BM_w = tf.concat([full_mainAnc_BM_w, mainAnc_BM_w], axis=1)
        full_mainAnc_BM_labels = tf.concat([full_mainAnc_BM_labels, mainAnc_BM_labels], axis=1)
        full_mainAnc_BM_scores = tf.concat([full_mainAnc_BM_scores, mainAnc_BM_scores], axis=1)
        full_mainAnc_BM_width = tf.concat([full_mainAnc_BM_width, Y_Fuse_label], axis=0)

    main_class_loss, main_loc_loss, main_conf_loss = \
        loss_function(full_mainAnc_class, full_mainAnc_conf,
                      full_mainAnc_xmin, full_mainAnc_xmax,
                      full_mainAnc_BM_x, full_mainAnc_BM_w,
                      full_mainAnc_BM_labels, full_mainAnc_BM_scores, config)
    main_fuse_loss = tf.reduce_mean(tf.square(full_mainAnc_width - full_mainAnc_BM_width))
    loss = main_class_loss + config.p_loc * main_loc_loss + config.p_conf * main_conf_loss + config.p_fuse * main_fuse_loss

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('main_class_loss', main_class_loss)
    tf.summary.scalar('main_loc_loss', main_loc_loss)
    tf.summary.scalar('main_conf_loss', main_conf_loss)
    tf.summary.scalar('main_fuse_loss', main_fuse_loss)

    trainable_variables = get_trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss, var_list=trainable_variables, global_step=global_step)

    return optimizer, loss, trainable_variables, main_class_loss, main_loc_loss, main_conf_loss, main_fuse_loss


def train_main(config):
    bsz = config.batch_size

    tf.set_random_seed(config.seed)
    X = tf.placeholder(tf.float32, shape=(bsz, config.input_steps, feature_dim))
    Y_label = tf.placeholder(tf.int32, [None, config.num_classes])
    Y_bbox = tf.placeholder(tf.float32, [None, 3])
    Index = tf.placeholder(tf.int32, [bsz + 1])
    LR = tf.placeholder(tf.float32)

    optimizer, loss, trainable_variables, main_class_loss, main_loc_loss, main_conf_loss, main_fuse_loss = \
        train_operation(X, Y_label, Y_bbox, Index, LR, config)

    model_saver = tf.train.Saver(var_list=trainable_variables, max_to_keep=6)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

    tf.global_variables_initializer().run()

    log_dir = cfg.log_dir + cfg.mode + cfg.pretrain_dataset + cfg.method
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = tf.summary.FileWriter(log_dir, sess.graph)
    merged = tf.summary.merge_all()

    # initialize parameters or restore from previous model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if os.listdir(models_dir) == [] or config.initialize:
        init_epoch = 0
        print ("Initializing Network")
    else:
        init_epoch = int(config.steps)
        restore_checkpoint_file = join(models_dir, 'model-ep-' + str(config.steps - 1))
        model_saver.restore(sess, restore_checkpoint_file)

    batch_train_dataX, batch_train_gt_label, batch_train_gt_info, batch_train_index = \
        get_train_data(config, cfg.mode, cfg.pretrain_dataset, True)
    num_batch_train = len(batch_train_dataX)

    for epoch in range(init_epoch, config.training_epochs):

        loss_info = []

        for idx in range(num_batch_train):
            feed_dict = {X: batch_train_dataX[idx],
                         Y_label: batch_train_gt_label[idx],
                         Y_bbox: batch_train_gt_info[idx],
                         Index: batch_train_index[idx],
                         LR: config.learning_rates[epoch]}
            _, out_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
            out_main_class_loss, out_main_loc_loss, out_main_conf_loss, out_main_fuse_loss, counter \
                = sess.run([main_class_loss, main_loc_loss, main_conf_loss, main_fuse_loss, global_step], feed_dict=feed_dict)

            [merged_] = sess.run([merged], feed_dict=feed_dict)
            writer.add_summary(merged_, counter)

            print('[Epoch/Idx][%2d/%3d] totoal_loss: %2.6f, main_class_loss: %2.6f, '
                  'main_loc_loss: %2.10f, main_conf_loss: %2.6f, main_fuse_loss: %2.6f' % (epoch, idx, out_loss,
                                                                                           out_main_class_loss,
                                                                                           out_main_loc_loss,
                                                                                           out_main_conf_loss,
                                                                                           out_main_fuse_loss))

            loss_info.append(out_loss)

        print ("Training epoch ", epoch, " loss: ", np.mean(loss_info))

        if epoch == config.training_epochs - 2 or epoch == config.training_epochs - 1 or epoch % 6 == 0:
        # if epoch == 5 or epoch == 10:
            model_saver.save(sess, models_file_prefix, global_step=epoch)


########################################### TEST ############################################

def test_operation(X, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    # MALs = main_anchor_layer(net)
    MALs, N_lsit, width_list = mx_fuse_anchor_layer(net, config)

    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')

        mainAnc_class, mainAnc_conf, mainAnc_rx, mainAnc_rw = anchor_box_adjust(mainAnc, config, ln, N_lsit[i])

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat([full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat([full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat([full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat([full_mainAnc_xmax, mainAnc_xmax], axis=1)

    full_mainAnc_class = tf.nn.softmax(full_mainAnc_class, dim=-1)
    return full_mainAnc_class, full_mainAnc_conf, full_mainAnc_xmin, full_mainAnc_xmax


def test_main(config):
    batch_dataX, batch_winInfo = get_test_data(config, cfg.mode, cfg.pretrain_dataset)

    X = tf.placeholder(tf.float32, shape=(config.batch_size, config.input_steps, feature_dim))

    anchors_class, anchors_conf, anchors_xmin, anchors_xmax = test_operation(X, config)

    # model_saver = tf.train.Saver()
    t_vars = tf.trainable_variables()
    model_saver = tf.train.Saver(t_vars)
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run()
    model_saver.restore(sess, test_checkpoint_file)

    batch_result_class = []
    batch_result_conf = []
    batch_result_xmin = []
    batch_result_xmax = []

    num_batch = len(batch_dataX)
    for idx in range(num_batch):
        out_anchors_class, out_anchors_conf, out_anchors_xmin, out_anchors_xmax = \
            sess.run([anchors_class, anchors_conf, anchors_xmin, anchors_xmax],
                     feed_dict={X: batch_dataX[idx]})
        batch_result_class.append(out_anchors_class)
        batch_result_conf.append(out_anchors_conf)
        batch_result_xmin.append(out_anchors_xmin * config.window_size)
        batch_result_xmax.append(out_anchors_xmax * config.window_size)

    outDf = pd.DataFrame(columns=config.outdf_columns)

    for i in range(num_batch):
        tmpDf = result_process(batch_winInfo, batch_result_class, batch_result_conf,
                               batch_result_xmin, batch_result_xmax, config, i)

        outDf = pd.concat([outDf, tmpDf])
    if config.save_predict_result:
        outDf.to_csv(predict_file, index=False)
    return outDf


if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    elapsed_time = 0

    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)

    if cfg.stage == 'train':
        train_main(config)
        elapsed_time = time.time() - start_time
    elif cfg.stage == 'test':
        df = test_main(config)
        elapsed_time = time.time() - start_time
        final_result_process(cfg.stage, cfg.pretrain_dataset, config, cfg.mode, cfg.method, '', df)
    elif cfg.stage == 'fuse':
        final_result_process(cfg.stage, cfg.pretrain_dataset, config, cfg.mode, cfg.method, cfg.method_temporal)
        elapsed_time = time.time() - start_time
    elif cfg.stage == 'train_test_fuse':
        train_main(config)
        elapsed_time = time.time() - start_time
        tf.reset_default_graph()
        df = test_main(config)
        final_result_process(cfg.stage, cfg.pretrain_dataset, config, cfg.mode, cfg.method, '', df)
    else:
        print ("No stage", cfg.stage, "Please choose a stage from train/test/fuse/train_test_fuse.")
    print ("Elapsed time:", elapsed_time, "start time:", start_time)
