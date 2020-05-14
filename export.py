import os
import sys
import logging
import datetime
import argparse

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import source_segmenter as drn
import numpy as np
from lib import _read_lists


raw_size = [256, 256, 3] # original raw input size
volume_size = [256, 256, 3] # volume size after processing
label_size = [256, 256, 1]

train_fid = "../Cardiac_4D/MRCT/PnpAda_release_data/train&val/mr_train_list"
val_fid = "../Cardiac_4D/MRCT/PnpAda_release_data/train&val/mr_val_list"
train_list = _read_lists(train_fid)
val_list =  _read_lists(val_fid)
path_prefix = '../Cardiac_4D/MRCT/PnpAda_release_data/train&val/'
train_list = [path_prefix + s for s in train_list]
val_list = [path_prefix + s for s in val_list]
saveDir = '../Cardiac_4D/MRCT/MR_valid/data'
saveDir_ = '../Cardiac_4D/MRCT/MR_valid/label'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
if not os.path.exists(saveDir_):
    os.makedirs(saveDir_)

decomp_feature = {
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

with tf.Session() as sess:
    queue = tf.train.string_input_producer(val_list, 
                                        num_epochs = None, shuffle = False)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(queue)
    parser = tf.parse_single_example(serialized_example, features = decomp_feature)
    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)

    data_vol = tf.reshape(data_vol, raw_size)
    label_vol = tf.reshape(label_vol, raw_size)
    data_vol = tf.slice(data_vol, [0,0,0], volume_size)
    label_vol = tf.slice(label_vol, [0,0,1], label_size)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(len(val_list)):
        example, l = sess.run([data_vol, label_vol])
        np.save(os.path.join(saveDir, str(i)+'.npy'), example)
        np.save(os.path.join(saveDir_, str(i)+'.npy'), l)
    coord.request_stop()
    coord.join(threads)