from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from pdb import set_trace
import bert
import input_builder
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re

import pandas as pd
MAX_SEQ_LENGTH = 128

os.environ['TFHUB_CACHE_DIR'] = '/home/djjindal/bert/script-learning'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def tokenize_if_small_enough(ds, sentences=True, no_context=True):
#     for d in ds:
    for i, d in zip(range(10000), ds):
        try:
            yield tokenize_dataset_dict(d, sentences, no_context)
        except AssertionError:
            continue

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
def get_token_ids(sentence, tokenizer, entity):
    tokens = []
    if type(sentence) == str:
        for orig_token in sentence.split(" "):
          temp = tokenizer.tokenize(orig_token)
          for t in temp:
            tokens.append(t)
    elif type(sentence) == tuple or type(sentence) == list:
        for svo in sentence:
          if svo is None:
              svo = entity
          for orig_token in svo.split(" "):
              temp = tokenizer.tokenize(orig_token)
              for t in temp:
                tokens.append(t)
    return tokens

    
"""# Make Data InputFeatures"""
#If candidates is list of strings, entity can be None
def convert_single_example2(event_chain, candidates, entity, label, max_seq_length, no_context,
                           tokenizer):
  tokens_e = []
  segment_ids_e = []
  input_id_list = []
  input_mask_list = []  
  segment_id_list = []
  tokens_e.append("[CLS]")
  segment_ids_e.append(0)
  
  # Fill Token Ids and Segment Ids from event chain
  if no_context != "True":
      for event in event_chain:
          tokens_e.extend(get_token_ids(event, tokenizer, entity))
          tokens_e.append("[SEP]")
      segment_ids_e = [0]*len(tokens_e)
    
  for candidate in candidates:
      tokens = []
      segment_ids = []
      tokens.extend(tokens_e)
      segment_ids.extend(segment_ids_e)
      candidate_tokens = get_token_ids(candidate, tokenizer, entity)
      tokens.extend(candidate_tokens)
      segment_ids.extend([1]*len(candidate_tokens))
            
      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)
        
      assert len(input_ids) <= max_seq_length
      assert len(input_mask) <= max_seq_length
      assert len(segment_ids) <= max_seq_length
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
      input_id_list.append(input_ids)
      input_mask_list.append(input_mask)
      segment_id_list.append(segment_ids)

  feature = input_builder.InputFeatures(
          input_ids=input_id_list,
          input_mask=input_mask_list,
          segment_ids=segment_id_list,
          label_id=label+1,
          is_real_example=True)
  return feature

def tokenize_dataset_dict(ec_dict, sentence, no_context):
  train_sents = ec_dict['sentences']
  train_triples = ec_dict['triples']
  candidates = ec_dict['candidates']
  correct_ending = ec_dict['correct']
  entity = ec_dict['entity']
  if sentence == "True":
      train_features = convert_single_example2(train_sents[:-1], candidates, entity, correct_ending, MAX_SEQ_LENGTH, no_context, tokenizer)
  else:
      train_features = convert_single_example2(train_triples[:-1], candidates, entity, correct_ending, MAX_SEQ_LENGTH, no_context, tokenizer)
  return train_features
