import tensorflow as tf
import tensorflow_hub as hub
from pdb import set_trace
import bert
import input_builder
import os
import numpy as np
import pandas as pd

MAX_SEQ_LENGTH = 256

os.environ['TFHUB_CACHE_DIR'] = '.'
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

CONCEPTNET_TABLE = pd.read_hdf('dataset/mini.h5')
CONCEPTNET_TABLE = CONCEPTNET_TABLE[CONCEPTNET_TABLE.index.map(lambda x: x.startswith('/c/en/'))]
CONCEPTNET_TABLE.index = CONCEPTNET_TABLE.index.map(lambda x: x.replace('/c/en/', ''))

def tokenize_if_small_enough(ds, sentences=True, no_context=True, is_neeg=False, conceptnet=False, semantic=False, input_size=10000):
    for i, d in zip(range(input_size), ds):
        try:
            yield tokenize_dataset_dict(d, sentence=sentences,
                                        no_context=no_context, is_neeg=is_neeg, conceptnet=conceptnet, semantic=semantic)
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
def get_token_ids(sentence, tokenizer, entity=None, is_neeg=False):
    tokens = []
    if is_neeg:
        sentence = filter(bool, sentence)
        for quad_token in sentence:
            for orig_token in quad_token.split(' '):
                for t in tokenizer.tokenize(orig_token):
                    tokens.append(t)
    elif type(sentence) == str:
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
    else:
        raise ValueError("Unexpected input in get_token_ids.")
    return tokens

    
"""# Make Data InputFeatures"""
#If candidates is list of strings, entity can be None
def convert_single_example2(tokenizer, event_chain, candidates, label, entity=None, max_seq_length=MAX_SEQ_LENGTH, no_context=False, is_neeg=False, conceptnet=False, semantic=False, pos_features=[], dep_features=[], max_sent=10, max_words=50):
  tokens_e = []
  segment_ids_e = []
  input_id_list = []
  input_mask_list = []  
  segment_id_list = []
  tokens_e.append("[CLS]")
  segment_ids_e.append(0)
  candidate_concept_vectors = [] 
  event_concept_vectors = []
  pos_features_vectors = []
  dep_features_vectors = []
  # Fill Token Ids and Segment Ids from event chain
  if not no_context:
      for event in event_chain:
          event_tokens = get_token_ids(event, tokenizer, entity, is_neeg)
          if conceptnet:
              vecs = [CONCEPTNET_TABLE.loc[tok] for tok in event_tokens if tok in CONCEPTNET_TABLE.index]
              event_concept_vectors.extend(vecs)
          tokens_e.extend(event_tokens)
          tokens_e.append("[SEP]")
      segment_ids_e = [0]*len(tokens_e)
    
  for candidate in candidates:
      tokens = []
      segment_ids = []
      tokens.extend(tokens_e)
      segment_ids.extend(segment_ids_e)
      candidate_tokens = get_token_ids(candidate, tokenizer, entity, is_neeg)
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
    
      if conceptnet:
            vecs = [CONCEPTNET_TABLE.loc[tok] for tok in candidate_tokens if tok in CONCEPTNET_TABLE.index]
            candidate_concept_vectors.extend(vecs)
  
  if conceptnet:
      if len(event_concept_vectors) > max_sent:
          event_concept_vectors = event_concept_vectors[:max_sent]
      else:
          event_concept_vectors.extend([np.zeros(300)] * (max_sent - len(candidate_concept_vectors)))
    
      event_concept_vectors = np.concatenate(event_concept_vectors)
      if len(candidate_concept_vectors) > max_sent:
          candidate_concept_vectors = candidate_concept_vectors[:max_sent]
      else:
          candidate_concept_vectors.extend([np.zeros(300)] * (max_sent - len(candidate_concept_vectors)))
      candidate_concept_vectors = np.concatenate(candidate_concept_vectors)
  if pos_features is not None and semantic:
      if len(pos_features) > max_sent:
          pos_features = pos_features[:max_sent]
          dep_features = dep_features[:max_sent]
      else:
          pos_features.extend([np.zeros(pos_features[0].shape)] * (max_sent - len(pos_features)))
          dep_features.extend([np.zeros(dep_features[0].shape)] * (max_sent - len(dep_features)))
      pos_features_vectors = np.concatenate(pos_features)
      dep_features_vectors = np.concatenate(dep_features)
      #pos_features_vectors = pos_features
      #dep_features_vectors = dep_features
  augmented = np.concatenate((pos_features_vectors, dep_features_vectors, event_concept_vectors, candidate_concept_vectors), axis=0)
  #print("Applying ConceptNet: {} Semantic{} Augmented Vector Shape", conceptnet, semantic, len(augmented))
  feature = input_builder.InputFeatures(
          input_ids=input_id_list,
          input_mask=input_mask_list,
          segment_ids=segment_id_list,
          label_id=label,
          augmented_vector=augmented,
          is_real_example=True)
  return feature

def tokenize_dataset_dict(ec_dict, sentence=True, no_context=False, is_neeg=False, conceptnet=False, semantic=False):
  if is_neeg:
      train_features = convert_single_example2(tokenizer, ec_dict['chain'], ec_dict['candidates'], ec_dict['correct'], 
                                               no_context=no_context, is_neeg=True, conceptnet=conceptnet)
      return train_features
  
  train_sents = ec_dict['sentences']
  candidates = ec_dict['candidates']
  correct_ending = ec_dict['correct']
  entity = ec_dict['entity']
  if sentence:
      train_features = convert_single_example2(tokenizer, train_sents, candidates, correct_ending,
                                               entity=entity, max_seq_length=MAX_SEQ_LENGTH, 
                                               no_context=no_context, is_neeg=is_neeg, conceptnet=conceptnet,
                                               semantic=semantic, pos_features=ec_dict['pos'], dep_features=ec_dict['dep'])
  else:
      train_triples = ec_dict['triples']
      train_features = convert_single_example2(tokenizer, train_triples, candidates, correct_ending,
                                               entity=entity, max_seq_length=MAX_SEQ_LENGTH,
                                               no_context=no_context, is_neeg=is_neeg, conceptnet=conceptnet)
  return train_features


def pad_to_max_of_max(ls_ls):
    inner_max_pad = max([max([np.array(t).shape for t in twe], key=lambda x:x[0])[0] 
                   for twe in ls_ls])
    inner_maxed = [np.array([np.pad(np.array(a), ((0, inner_max_pad-np.array(a).shape[0]), (0, 0)), mode='constant') for a in ls]) for ls in ls_ls]
    outer_max = max((e.shape[0] for e in inner_maxed))
    return np.array([np.pad(e, ((0, outer_max-e.shape[0]), (0, 0), (0, 0)), mode='constant') for e in inner_maxed])
    
def pad_to_max(ls):
    ls = [np.array(e) for e in ls]
    max_pad = max((e.shape[0] for e in ls))
    return np.array([np.pad(e, ((0, max_pad-e.shape[0]), (0, 0)), mode='constant') for e in ls])
