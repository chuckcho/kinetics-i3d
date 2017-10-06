import tensorflow as tf
import numpy as np
import os
import threading
from config import *
# import train code here

class InputPipeLine(object):
  def __init__(self, num_frames, batch_size, frame_stride):
    self.num_frames = num_frames
    self.batch_size = batch_size
    self.stride = frame_stride
    self.cls_dict = {}
    folders = np.sort([f for f in os.listdir(FRAME_DATA_PATH) if f.startswith('v')])
    l = 0
    for folder in folders:
      cls_name = folder.split('_')[1]
      if not cls_name in self.cls_dict:
        self.cls_dict[cls_name] = l
        l += 1
    # placeholders
    self.rgb = tf.placeholder(tf.string, shape=[self.num_frames])
    self.flow_x = tf.placeholder(tf.string, shape=[self.num_frames])
    self.flow_y = tf.placeholder(tf.string, shape=[self.num_frames])
    self.label = tf.placeholder(tf.int32)

    self.queue = tf.FIFOQueue(capacity=QUEUE_CAPACITY, dtypes=[tf.string, tf.string, tf.string, tf.int32], shapes=[[self.num_frames],[self.num_frames],[self.num_frames],[]])


  def _enqueue(self, sess, enqueue_op):
    folders = [folder for folder in os.listdir(FRAME_DATA_PATH) if folder.startswith('v')]
    while True:
      index = np.random.randint(0, len(folders))
      prefix = os.path.join(FRAME_DATA_PATH, folders[index])
      cls_name = folders[index].split('_')[1]
      sorted_list = np.sort(os.listdir(prefix))
      imgs = [os.path.join(prefix, img) for img in sorted_list if img.startswith('img')]
      flow_xs = [os.path.join(prefix, flow) for flow in sorted_list if flow.startswith('flow_x')]
      flow_ys = [os.path.join(prefix, flow) for flow in sorted_list if flow.startswith('flow_y')]
      assert len(imgs) == len(flow_xs)
      assert len(imgs) == len(flow_ys)
      if self.num_frames <= len(imgs):
        begin = np.random.randint(0, len(imgs) - self.num_frames + 1)
      else:
        begin = 0
        ori_len = len(imgs)
        while len(imgs) < self.num_frames:
          for i in range(0, ori_len, self.stride):
            imgs.append(imgs[i])
            flow_xs.append(flow_xs[i])
            flow_ys.append(flow_ys[i])
            if len(imgs) == self.num_frames:
              break

      imgs_out = imgs[begin:begin + self.num_frames]
      flow_xs_out = flow_xs[begin:begin + self.num_frames]
      flow_ys_out = flow_ys[begin:begin + self.num_frames]
      sess.run(enqueue_op, {self.rgb: imgs_out, self.flow_x: flow_xs_out, self.flow_y: flow_ys_out, self.label: self.cls_dict[cls_name]})

  def start(self, sess):
    enqueue_op = self.queue.enqueue([self.rgb, self.flow_x, self.flow_y, self.label])
    enqueue_thread = threading.Thread(target=self._enqueue, args=[sess, enqueue_op])
    enqueue_thread.daemon = True
    enqueue_thread.start()
    # start pipeline before start tf queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord, threads

  def get_batch(self):
    item = self.queue.dequeue()
    rgb_frames = []
    flow_x_frames = []
    flow_y_frames = []
    for i in range(self.num_frames):
      rgb_frames.append(tf.image.decode_jpeg(tf.read_file(item[0][i]), channels=3))
      flow_x_frames.append(tf.image.decode_jpeg(tf.read_file(item[1][i]), channels=1))
      flow_y_frames.append(tf.image.decode_jpeg(tf.read_file(item[2][i]), channels=1))

    output_rgb = tf.stack(rgb_frames, axis=0)
    tmp_flow_x = tf.stack(flow_x_frames, axis=0)
    tmp_flow_y = tf.stack(flow_y_frames, axis=0)
    output_flow = tf.concat([tmp_flow_x, tmp_flow_y], axis=3)

    # random crop
    rgb_flow_concat = tf.concat([output_rgb, output_flow], axis=3)
    crop_concat = tf.random_crop(rgb_flow_concat, [int(self.num_frames), CROP_SIZE, CROP_SIZE, 5])
    output_rgb = crop_concat[:,:,:,:3]
    output_flow = crop_concat[:,:,:,3:]

    # rescale
    output_rgb = tf.cast(output_rgb, tf.float32)
    output_flow = tf.cast(output_flow, tf.float32)
    output_rgb = output_rgb * 2 / 255.0 - 1
    output_flow = output_flow * 2 / 256.0 - 1

    label = tf.cast(item[3], tf.int32)
    rgbs, flows, labels = tf.train.batch([output_rgb, output_flow, label], batch_size=self.batch_size)
    return rgbs, flows, labels

if __name__ == '__main__':
  with tf.Graph().as_default() as g:
    pipeline = InputPipeLine(20, 10, 1) # (NUM_FRAMES, BATCH_SIZE, FRAME_STRIDE)
    rgbs, flows, labels = pipeline.get_batch()

    with tf.Session() as sess:
      coord, threads = pipeline.start(sess) # start input pipeline with sess

      rgbs_res, flows_res = sess.run([rgbs, flows])
      # print 'RGB', rgbs_res[0].min(), rgbs_res[0].max() 
      # print 'flow', flows_res[0].min(), flows_res[0].max()
      print rgbs_res.shape, flows_res.shape
      coord.request_stop()
      coord.join(threads)
