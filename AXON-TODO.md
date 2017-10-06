To-do's
-------
Priority 1
- random horizonal flip to be implemented as data augmentation (2.5 in https://arxiv.org/pdf/1705.07750.pdf)
- add (clip-level) validation for every Nth step
- calculate number of examples for whole training data so that we know 1 epoch = ? iterations
- add summaries of training/val loss, val accuracy, etc.  as training log (viewed via tensorboard)

Priority 2
- when training is interrupted (Catching KeyboardInterrupt), save checkpoint before exitting
- multi-gpu
- be able to specify training/validation data with a file (a list of image/flow directories)
- evaluation script (input: a text file that lists all test video directories, ckpt dir, output: video-level accuracy)

Priority 3
- try https://github.com/uber/horovod
