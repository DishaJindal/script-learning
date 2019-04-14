from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from IPython.core.debugger import set_trace
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
#from google.colab import drive
#drive.mount('/content/gdrive/')

os.environ['TFHUB_CACHE_DIR'] = '/home/djjindal/bert/script-learning'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  #set_trace()
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

"""# Make Data InputFeatures"""

def convert_single_example(ex_index, example, candidates, label, max_seq_length,
                           tokenizer):
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  
  i = 0
  for line in example:
    for orig_token in line.split(" "):
      temp = tokenizer.tokenize(orig_token)
      for t in temp:
        tokens.append(t)
        segment_ids.append(i)
    tokens.append("[SEP]")
    segment_ids.append(i)
    i = i+1

#   i = 0
  for line in candidates:
    for orig_token in line.split(" "):
      temp = tokenizer.tokenize(orig_token)
      for t in temp:
        tokens.append(t)
        segment_ids.append(i)
    
    tokens.append("[SEP]")
    segment_ids.append(i)
    i = i+1
  
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  input_mask = [1] * len(input_ids)

#   print(len(input_mask), len(segment_ids))
#   print(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
#   print("input_ids", input_ids)
#   print("input_mask", input_mask)
#   print("segment_ids", segment_ids)
#   print("label", label-1)
  
  feature = run_classifier.InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label-1,
      is_real_example=True)
  return feature

def convert_examples_to_features(examples, candidates, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  i = 0
  for (example) in (examples):
    feature = convert_single_example(i, example, candidates[i], label_list[i], max_seq_length, tokenizer)
    features.append(feature)
    i = i+1
  return features

import pandas as pd
MAX_SEQ_LENGTH = 128


def createData(file):
  data = pd.read_csv(file)
  train = (data[['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4']]).values.tolist()
  candidates = (data[['RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2']]).values.tolist()
  label_lists = (data[['AnswerRightEnding']]).values.tolist()

  label_list = []
  for label in label_lists:
    label_list.append(label[0])

  MAX_SEQ_LENGTH = 128
  train_features = convert_examples_to_features(train, candidates,label_list, MAX_SEQ_LENGTH, tokenizer)
  return train_features
