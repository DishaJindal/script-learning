import tensorflow as tf
#tf.enable_eager_execution()
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow_hub as hub
from datetime import datetime
from pdb import set_trace
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
from model import *
from prepare_data import *
import prepare_data
from read import *
import numpy as np

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True,
               augmented_vector=None
              ):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    self.augmented_vector = augmented_vector
    
    def __repr__(self):
        return str(self.__dict__)
    
    def __str__(self):
        return "Class %d, tokens"%self.label_id + str(self.input_ids)
    
    def __unicode__(self):
        return "test"
    
def input_fn_builder(features, seq_length, is_training, drop_remainder, candidates):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []
  all_augmented_vector = []


  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)
    all_augmented_vector.append(feature.augmented_vector)
  
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, candidates, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, candidates, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, candidates, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        "augmented_vector":
            tf.constant(
                np.array(all_augmented_vector), 
                shape=[num_examples, all_augmented_vector[0].shape[0]], 
                dtype=tf.float32)
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn