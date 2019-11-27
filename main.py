import os
import numpy as np

from model import DCGAN
from utils import pp

import tensorflow as tf

# arguments parsers
flags = tf.app.flags

flags.DEFINE_float("res_loss_w", 0.9, "Weight of the residula loss in the total loss")


# Train
flags.DEFINE_boolean("train", False, "Do you want to train the model")
flags.DEFINE_integer("epoch", 1, "Epoch to train [5]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("train_size", 100000, "The number of images to use in train set")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

# Test
flags.DEFINE_boolean("test_qual", False, "True for anomaly test in test directory, not anomaly test [False]")
flags.DEFINE_boolean("test_quant", False, "Run Quantitative tests")
flags.DEFINE_integer("test_epoch", 300, "Epoch for latent mapping in anomaly detection to train [100]")
flags.DEFINE_integer("test_batch_size", 1, "The size of test batch images in anomaly detection to [1]")
flags.DEFINE_float("test_learning_rate", 0.001, "Learning rate for finding latent variable z [0.05]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")


# Other
flags.DEFINE_string("dataset", "moons", "Name of dataset to use")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("test_dir", "test_data", "Directory name to load the anomaly detstion result [test_data]")
flags.DEFINE_string("test_result_dir", "test_result", "Directory name to save the anomaly test result"
                                                      " [test_data/test_result]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint dir")


FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
  if not os.path.exists('samples'):
    os.makedirs('samples')

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  #run_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=run_config) as sess:

    dcgan = DCGAN(
      sess,
      input_width=100,
      input_height=100,
      output_width=100,
      output_height=100,
      batch_size=FLAGS.batch_size,
      test_batch_size=FLAGS.test_batch_size,
      sample_num=FLAGS.batch_size,
      z_dim=FLAGS.generate_test_images,
      dataset_name=FLAGS.dataset,
      input_fname_pattern=FLAGS.input_fname_pattern,
      crop=FLAGS.crop,
      checkpoint_dir='checkpoint',
      sample_dir='samples',
      test_dir=FLAGS.test_dir)

    if FLAGS.train:
      print(f"TRAIN PHASE")
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
    
    if FLAGS.test_qual:
      print(f"TEST PHASE (QUALITATIVE)")
      os.makedirs(f'./test_result/{FLAGS.dataset}', exist_ok=True)
      dcgan.anomaly_detector(res_loss_w=FLAGS.res_loss_w)
      assert len(dcgan.test_data_names) > 0

      for idx in range(len(dcgan.test_data_names)):
        test_input = np.expand_dims(dcgan.test_data[idx],axis=0)
        test_name = dcgan.test_data_names[idx]
        # Returns the anomality score (res_loss + detector_loss) and the res_loss
        sample, error, ano_score, res_loss = dcgan.train_anomaly_detector(FLAGS, test_input, test_name)


if __name__ == '__main__':
  tf.app.run()
