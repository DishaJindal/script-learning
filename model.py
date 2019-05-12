from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from pdb import set_trace
import bert
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
from model import *
from prepare_data import *
from sklearn.metrics import classification_report

#os.environ['TFHUB_CACHE_DIR'] = '/home/djjindal/bert/script-learning'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_model3(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels, augmenting_vectors=None):

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
    
  for i in range(0,num_labels):
    input_ids_c = input_ids[:,i,:]
    input_mask_c = input_mask[:,i,:]
    segment_ids_c = segment_ids[:,i,:]
    bert_inputs = dict(
          input_ids=input_ids_c,
          input_mask=input_mask_c,
          segment_ids=segment_ids_c)
    bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer_temp = bert_outputs["pooled_output"]
    if augmenting_vectors is not None:
        output_layer_temp = tf.concat([output_layer_temp, augmenting_vectors[i]], axis=1) 
    
    if i == 0:
        output_layer = output_layer_temp
    else:
        output_layer = tf.concat([output_layer, output_layer_temp], axis=1) 
  
  hidden_size = output_layer.shape[-1].value
#   set_trace()

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    if not is_predicting:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)

def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    augmenting_vectors = features.get('augmenting_vectors', None)

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:
      (loss, predicted_labels, log_probs) = create_model3(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, augmenting_vectors=augmenting_vectors)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn_multi(label_ids, predicted_labels, name):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        return {name: accuracy}
      
      if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("Train_Loss", loss)
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss, train_op=train_op, eval_metric_ops=metric_fn_multi(label_ids, predicted_labels, "train_accuracy"))
      else:
        tf.summary.scalar("Eval_Loss", loss)
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss, eval_metric_ops=metric_fn_multi(label_ids, predicted_labels, "eval_accuracy"))
    else:
      (predicted_labels, log_probs) = create_model3(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, augmenting_vectors=augmenting_vectors)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }

      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn
