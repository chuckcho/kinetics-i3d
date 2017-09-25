from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400
_BATCH_SIZE = 100

_DROPOUT_KEEP_PROB = 0.5
_MAX_ITER = 100000

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

def train():
  rgb_inputs, flow_inputs, labels = inputs(_BATCH_SIZE)

  with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
      _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
      rgb_input, is_training=True, dropout_keep_prob=_DROPOUT_KEEP_PROB)
  rgb_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
      rgb_variable_map[variable.name.replace(':0', '')] = variable
  rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(
        _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    flow_logits, _ = flow_model(
        flow_input, is_training=True, dropout_keep_prob=_DROPOUT_KEEP_PROB)
  flow_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Flow':
      flow_variable_map[variable.name.replace(':0', '')] = variable
  flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  model_logits = rgb_logits + flow_logits
  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model_logits))
  lr = 0.01
  opt = tf.train.GradientDescentOptimizer(lr)
  train_op = opt.minimize(loss)

  # saver for fine tuning
  saver = tf.train.Saver(max_to_keep=10)
  ckpt_path = './tmp/ckpt'
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
      rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])

    # we're going to use queue in inputs(), so we need to start queue runners 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    it = 0
    while it < _MAX_ITER:
      sess.run(train_op)
      if i % 1000 == 0:
        loss_val = sess.run(loss)
        print 'step %d, loss = %.3f' % (i, loss_val)
        if i > 0:
          saver.save(sess, ckpt_path + '/model_ckpt', i)
      i += 1

if __name__ == '__main__':
  train()