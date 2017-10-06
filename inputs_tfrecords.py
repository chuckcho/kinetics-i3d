import tensorflow as tf
import os
import numpy as np
# The training data should be organized as follows:
# train/
#   class1/
#     rgb/
#       rgb1.npy
#       rgb2.npy
#       ...
#     flow/
#       flow1.npy
#       flow2.npy
#       ...
#     ...
#   class2/
#     ...
#   ... 

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 79
_TRAIN_DATA_PATH = 'data'


def _float_feature(ndarr):
  return tf.train.Feature(float_list=tf.train.FloatList(value=ndarr.flatten().tolist()))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_npy_to_tfrecords():
  writer = tf.python_io.TFRecordWriter('train.tfrecords')
  i = 0
  for folder in os.listdir(_TRAIN_DATA_PATH):
    if folder == '.DS_Store':
      continue

    # rgb_files = ['data/v_CricketShot_g04_c01_rgb.npy']
    # flow_files = ['data/v_CricketShot_g04_c01_flow.npy']
    rgb_files = os.listdir(_TRAIN_DATA_PATH + '/' + folder + '/rgb')
    flow_files = os.listdir(_TRAIN_DATA_PATH + '/' + folder + '/flow')
    num_files = len(rgb_files)

    for j in range(num_files):
      rgb = np.load(rgb_files[j])
      flow = np.load(flow_files[j])
      
      feature = {
        'rgb': _float_feature(rgb),
        'flow': _float_feature(flow),
        'label': _int_feature(i)
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
    i += 1
  writer.close()

def inputs(batch_size):
  feature = {
    'rgb': tf.FixedLenFeature([_SAMPLE_VIDEO_FRAMES * _IMAGE_SIZE * _IMAGE_SIZE * 3], tf.float32),
    'flow': tf.FixedLenFeature([_SAMPLE_VIDEO_FRAMES * _IMAGE_SIZE * _IMAGE_SIZE * 2], tf.float32),
    'label': tf.FixedLenFeature([1], tf.int64)
  }

  data_queue = tf.train.string_input_producer(['train.tfrecords'])
  reader = tf.TFRecordReader()
  _, data = reader.read(data_queue)
  features = tf.parse_single_example(data, features=feature)
  print features
  rgb = tf.reshape(features['rgb'], [_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
  flow = tf.reshape(features['flow'], [_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2])
  label = tf.cast(features['label'], tf.int32)
  
  rgbs, flows, labels = tf.train.shuffle_batch([rgb, flow, label], 
                                       batch_size=batch_size, 
                                       num_threads=4,
                                       capacity=3 + batch_size,
                                       min_after_dequeue=1)
  return rgbs, flows, labels

# if __name__ == '__main__':
#   convert_npy_to_tfrecords()

#   r,f,l = inputs(1)
#   with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     img, flow, label = sess.run([r, f, l])
#     print img
