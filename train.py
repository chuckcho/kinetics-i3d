from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
import tensorflow as tf
from inputs_new import *

import i3d
from config import *

# build the model
def inference(rgb_inputs, flow_inputs):
  with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
      NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
      rgb_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(
        NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    flow_logits, _ = flow_model(
        flow_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  return rgb_logits, flow_logits

# restore the pretrained weights, except for the last layer
def restore():
  # rgb
  rgb_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
      if 'Logits' in variable.name: # skip the last layer
        continue
      rgb_variable_map[variable.name.replace(':0', '')] = variable
  rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
  # flow
  flow_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Flow':
      if 'Logits' in variable.name: # skip the last layer
        continue
      flow_variable_map[variable.name.replace(':0', '')] = variable
  flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  return rgb_saver, flow_saver


def loss(logits, labels):
  return tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))

def train(loss):
  lr = LR
  opt = tf.train.GradientDescentOptimizer(lr)
  train_op = opt.minimize(loss)
  return train_op

if __name__ == '__main__':
  pipeline = InputPipeLine(NUM_FRAMES, BATCH_SIZE, FRAME_STRIDE)

  rgbs, flows, labels = pipeline.get_batch()
  rgb_logits, flow_logits = inference(rgbs, flows)
  rgb_saver, flow_saver = restore()
  total_loss = loss(rgb_logits + flow_logits, labels)
  train_op = train(total_loss)

  # saver for fine tuning
  if not os.path.exists(TMPDIR):
    os.mkdir(TMPDIR)
  saver = tf.train.Saver(max_to_keep=SAVER_MAX_TO_KEEP)
  ckpt_path = os.path.join(TMPDIR, 'ckpt')
  if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring from:', ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
      print 'No checkpoint file found, restoring pretrained weights...'
      rgb_saver.restore(sess, CHECKPOINT_PATHS['rgb_imagenet'])
      flow_saver.restore(sess, CHECKPOINT_PATHS['flow_imagenet'])
      print 'Restore Complete.'

    coord, threads = pipeline.start(sess)

    it = 0
    while it < MAX_ITER and not coord.should_stop():
      if it % DISPLAY_ITER == 0:
        _, loss_val = sess.run([train_op, total_loss])
        print 'step %d, loss = %.3f' % (it, loss_val)
        # if it > 0:
        # saver.save(sess, ckpt_path + '/model_ckpt', it)
      else:
        sess.run(train_op)
      it += 1
    coord.request_stop()
    coord.join(threads)
