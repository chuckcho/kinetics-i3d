#!/usr/bin/env python

FRAME_DATA_PATH = '/media/6TB/Videos/UCF-101-frames-mini'
CROP_SIZE = 224
NUM_CLASSES = 101
BATCH_SIZE = 2
NUM_FRAMES = 64
FRAME_STRIDE = NUM_FRAMES
QUEUE_CAPACITY = 32

DROPOUT_KEEP_PROB = 0.5
MAX_ITER = 100000

CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

LR = 0.01 # can change it to exponentially decay with global steps
TMPDIR = './tmp'

DISPLAY_ITER = 10
SAVER_MAX_TO_KEEP = 10
