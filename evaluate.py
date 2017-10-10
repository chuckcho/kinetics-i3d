#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
import i3d
from inputs_new import *
from config import *
import argparse

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


def evaluate(input_file, ckpt_dir):
  with tf.Graph().as_default() as g:
    pipeline = InputPipeLine(input_file, num_epochs=1)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print 'Restoring from:', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print 'error restoring ckpt...'
        return
      coord, threads = pipeline.start(sess)
      rgbs, flows, labels = pipeline.get_batch(train=False)
      rgb_logits, flow_logits = inference(rgbs, flows)
      model_logits = rgb_logits + flow_logits
      top_k_op = tf.nn.in_top_k(model_logits, labels, 1)
      true_count = 0
      try: 
        while not coord.should_stop():
          true_count += np.sum(sess.run(top_k_op))
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads)

      print 'eval accuracy: %.3f' % (true_count / pipeline.num_inputs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file')
  parser.add_argument('--ckpt_dir', required=True)
  args = parser.parse_args()  
  evaluate(args.input_file, args.ckpt_dir)