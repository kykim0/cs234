"""Plot graphs."""

import collections
import glob
import os

from absl import app
from absl import flags
from absl import logging

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('paths', None, 'CSV of event file paths.')
flags.DEFINE_string('outdir', '/tmp', 'Output directory.')
flags.DEFINE_string('names', None, 'Names to show in legend.')
flags.DEFINE_string('metric_tag', 'Metrics/AverageReturn', 'Metric name.')
flags.DEFINE_string('title', None, 'Title of the plot.')

MAX_STEPS = 1000000


def load_tfevents_file(path):
  """Loads a single TensorFlow event file."""
  metrics = []
  for event in tf.train.summary_iterator(path):
    if event.step > MAX_STEPS: continue
    for value in event.summary.value:
      if value.tag != FLAGS.metric_tag: continue
      # TensorFlow 2.0 stores raw tensor content in binary format.
      tensor_dtype = tf.dtypes.as_dtype(value.tensor.dtype)
      contents = np.frombuffer(
          value.tensor.tensor_content, tensor_dtype.as_numpy_dtype)
      if contents:
        metrics.append((event.step, contents[0]))
  return metrics


def plot(metrics_set, names, outdir):
  """Creates a plot."""
  plt.figure()
  plt.title(FLAGS.title)
  plt.xlabel('Step')
  plt.ylabel('Average Return')
  for key, metrics in metrics_set.items():
    xs, ys = [], []
    for step, value in metrics:
      xs.append(step)
      ys.append(value)
    df = pd.DataFrame(list(zip(xs, ys)), columns=['x', 'y'])
    plt.plot(df.x, df.y.rolling(window=5).mean())
  plt.legend(names, loc='lower right')
  plt.savefig(os.path.join(FLAGS.outdir, 'plot.png'))


def main(_):
  results = {}
  file_paths = FLAGS.paths.split(',')
  names = file_paths if not FLAGS.names else FLAGS.names.split(',')
  for idx, path in enumerate(file_paths):
    for filename in glob.glob(
        os.path.join(path, 'eval', 'events.out.tfevents.*')):
      results[names[idx]] = load_tfevents_file(filename)
  plot(results, names, FLAGS.outdir)


if __name__ == '__main__':
  app.run(main)
