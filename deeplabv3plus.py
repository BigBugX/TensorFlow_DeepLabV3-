__author__ = 'Will@PCVG'
# Utils used with tensorflow implemetation

from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
import copy
import functools

from ops_dup import *

import tensorflow as tf
import numpy as np

import TensorflowUtils_plus as utils
import datetime
from portrait_plus import BatchDatset, TestDataset
from PIL import Image
from scipy import misc
import scipy.io as scio
import os
from tensorflow.python import pywrap_tensorflow

from PIL import Image
import cv2
import torch

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
is_train = True

cur_path = os.getcwd()
model_path = cur_path + "\\" + 'new_mob718'

prefix = "module/features/"

def dlv3p_718(weights, image):
    exp = 6
    net = {}
    current = image

    # MobileNet V2 1/8

    ## 'Conv' { in_ch:3 out_ch:32 kernel:3 stride:2 }
    curfix = prefix + '0/'
    Conv_w = weights[curfix + '0/weight']
    Conv_w = Conv_w.transpose((2,3,1,0))
    bn_w = weights[curfix + '1/weight']
    bn_b = weights[curfix + '1/bias']
    bn_m = weights[curfix + '1/running_mean']
    bn_v = weights[curfix + '1/running_var']

    current = conv2d_head_oct_dl(current, Conv_w, bn_w, bn_b, bn_m, bn_v, strides=2, name='Conv')

    ## 'block1' 
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv', invres_idx=1)

    ## 'block2'
    current = invres_oct_dl(current, weights, strides=2, name='expanded_conv1', invres_idx=2)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv2', invres_idx=3)
    net['low_features'] = current
    
    ## 'block3'
    current = invres_oct_dl(current, weights, strides=2, name='expanded_conv3', invres_idx=4)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv4', invres_idx=5)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv5', invres_idx=6)

    ## 'block4'
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv6', invres_idx=7)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv7', invres_idx=8)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv8', invres_idx=9)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv9', invres_idx=10)

    ## 'block5'
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv10', invres_idx=11)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv11', invres_idx=12)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv12', invres_idx=13)

    ## 'block6'
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv13', invres_idx=14)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv14', invres_idx=15)
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv15', invres_idx=16)
    
    ## 'block7'
    current = invres_oct_dl(current, weights, strides=1, name='expanded_conv16', invres_idx=17)
    net['high_features'] = current


    return net

"""  ***testing codes***
weights_path = os.getcwd() + '\\models\\' + 'new_mob718'
weights = torch.load(weights_path)
image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="input_image")

with tf.variable_scope("inference"):
    image_net = dlv3p_718(weights, image)
    print('good')
"""

def inference(image):
    weights_path = os.getcwd() + "\\models\\" + "new_mob718"
    weights = torch.load(weights_path)

    with tf.variable_scope("inference"):
        image_net = dlv3p_718(weights, image)
        low_level_feat = image_net['low_features']
        high_level_feat = image_net['high_features']
        high_shape = high_level_feat.get_shape().as_list()

        x_aspp = utils.aspp_dl(high_level_feat, 320, 256)
        x_ = utils.global_avgp_dl(high_level_feat, 320, 256, name='global_avgp')
        x_ = utils.upsample_dl(x_, high_shape[1], high_shape[2])
        high_level_feat = tf.concat([x_aspp, x_], 3, name="fuse_oct_1") # 1/8 feature maps

        high_level_feat = utils.aspp_conv2d_dl(high_level_feat, 1280, 256, 1, name='conv_h')
        high_level_feat = utils.aspp_bn_dl(high_level_feat, name="bn_h")
        high_shape = high_level_feat.get_shape().as_list()
        high_level_feat = utils.upsample_dl(high_level_feat, 2*(high_shape[1]), 2*(high_shape[2])) # 1/4 feature_maps

        low_level_feat = utils.aspp_conv2d_dl(low_level_feat, 24, 48, 1, name="conv_l")
        low_level_feat = utils.aspp_bn_dl(low_level_feat, name="bn_l")

        high_level_feat = tf.concat([high_level_feat, low_level_feat], 3)

        high_level_feat = utils.aspp_conv2d_dl(high_level_feat, 304, 20, 1, name="conv_pred")
        high_shape = high_level_feat.get_shape().as_list()
        high_level_feat = utils.upsample_dl(high_level_feat, 4*high_shape[1], 4*high_shape[2])

        annotation_pred = tf.argmax(high_level_feat, dimension=3, name='prediction')        
        # annotation_pred = high_level_feat
    return tf.expand_dims(annotation_pred, dim=3), high_level_feat

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def get_batch2(batch_size, img_lst, label_lst, org_idx):    
    img_, label_ = get_batch(img_lst[org_idx], label_lst[org_idx])

    for i in range(1, batch_size):
        img_b, label_b = get_batch(img_lst[i], label_lst[i])
        img_ = np.concatenate((img_, img_b), axis=0)
        label_ = np.concatenate((label_, label_b), axis=0)

    return img_, label_

def get_batch(img_path, label_path):
    img_ = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)/255.0
    # img_ = cv2.imread(img_path)
    # img_ = img_.astype(np.float32)/255.0
    img_[:,:,0] = img_[:,:,0] - 0.4914
    img_[:,:,1] = img_[:,:,1] - 0.4822
    img_[:,:,2] = img_[:,:,2] - 0.4465
    img_ = cv2.resize(img_, (256, 256))

    img_output = np.expand_dims(img_, axis=0)
    label_ = np.array(scio.loadmat(label_path)["GTcls"][0]['Segmentation'][0]).astype(np.float32)
    label_ = cv2.resize(label_, (256, 256))
    label_tmp = np.expand_dims(label_, axis=0)
    label_min = label_tmp.min()
    label_max = label_tmp.max()
    label_scaled = (label_tmp - label_min) / (label_max - label_min)
    label_output = np.expand_dims(label_scaled, axis=3)
    return img_output, label_output

def get_dataset():
    base_dir = "C:\\Users\\XING Wei Will\\Desktop\\integrated\\data3\\benchmark\\benchmark_RELEASE\\dataset\\"
    _dataset_dir = base_dir
    _image_dir = _dataset_dir + 'img'
    _cat_dir = _dataset_dir + 'cls'

    im_ids = []
    images = []
    categories = []

    with open(os.path.join(_dataset_dir, 'train.txt'), "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        _image = os.path.join(_image_dir, line + '.jpg')
        _categ = os.path.join(_cat_dir, line + '.mat')
        assert os.path.isfile(_image)
        assert os.path.isfile(_categ)
        im_ids.append(line)
        images.append(_image)
        categories.append(_categ)

    assert (len(images) == len(categories))

    return im_ids, images, categories

def get_test_batch(img_path):
    img_ = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)/255.0

    img_[:,:,0] = img_[:,:,0] - 0.4914
    img_[:,:,1] = img_[:,:,1] - 0.4822
    img_[:,:,2] = img_[:,:,2] - 0.4465

    org_shape = img_.shape
    img_ = cv2.resize(img_, (256, 256))
    img_output = np.expand_dims(img_, axis=0)
    return org_shape, img_output

def get_test_dataset():
    base_dir = "C:\\Users\\XING Wei Will\\Desktop\\integrated\\data3\\benchmark\\benchmark_RELEASE\\dataset\\"
    _dataset_dir = base_dir
    _image_dir = _dataset_dir + 'img'

    im_ids = []
    images = []

    with open(os.path.join(_dataset_dir, 'val.txt'), "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        _image = os.path.join(_image_dir, line + '.jpg')
        assert os.path.isfile(_image)
        im_ids.append(line)
        images.append(_image)
    assert (len(images) == len(im_ids))
    return im_ids, images

def get_voc_set():
    base_dir = "C:\\Users\\XING Wei Will\\Desktop\\integrated\\data3\\VOCdevkit\\VOC2012"


def main(argv=None):

    batch_size = 16
    #keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, 256, 256, 1], name="annotations")

    pred_annotation, logits = inference(image)
    # sft = tf.nn.softmax(logits)

    # logits = tf.expand_dims(logits, 3)
    logits = tf.to_float(logits)
    # annotation = tf.to_float(annotation)
 
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                        labels = tf.squeeze(annotation,3),
                                                                        name="entropy")))

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    # get training set
    im_ids = []
    images = []
    labels = []

    im_ids, images, labels = get_dataset()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(0,len(im_ids),batch_size):
            img_, label_ = get_batch2(batch_size, images, labels, i)

            """
            img_, label_ = get_batch(images[i], labels[i])
            assert not np.any(np.isnan(img_))
            assert not np.any(np.isnan(label_))
            """

            trloss = 0.0
            feed_dict = {image: img_, annotation: label_}
            _, rloss = sess.run([train_op, loss], feed_dict=feed_dict)
            trloss += rloss
            
            print("Step: %d, Train_loss:%f" % (i, rloss / batch_size))
            """
            if i % 10 == 0:
                print("Step: %d, Train_loss:%f" % (i, trloss / 10))
                trloss = 0.0
            """
            
        saver.save(sess, FLAGS.logs_dir + "dlv3p_718.ckpt", i)


def pred():

    image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, 256, 256, 1], name="annotations")

    im_ids = []
    images = []

    im_ids, images= get_test_dataset()

    pred_annotation, logits = inference(image)
    stf = tf.nn.softmax(logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        for i in range(0, 1):
            org_shape, img_ = get_test_batch(images[i])
            assert not np.any(np.isnan(img_))
            feed_dict = {image: img_}
            preds = sess.run(pred_annotation, feed_dict)
            new_shape = (org_shape[1], org_shape[0])
            preds = preds.astype(np.float32)
            preds = cv2.resize(preds[0,:,:,:], new_shape)
            outputimg = np.zeros([org_shape[0], org_shape[1], 3])
            outputimg[:,:,0] = preds
            outputimg[:,:,1] = preds
            outputimg[:,:,2] = preds
            misc.imsave('testpreds0924_%d.jpg'%i, outputimg)



if __name__ == '__main__':
    # tf.app.run()
    pred()
