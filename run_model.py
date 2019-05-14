import argparse
import os

parser = argparse.ArgumentParser(description='ScriptLearning')
parser.add_argument('mode', help="[train, predict]")
parser.add_argument('--data', type=str, default="dataset/gw_extractions_no_rep_no_fin.pickle")
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--sentence', default=False, action='store_true')
parser.add_argument('--output_dir',type=str, default="output")
parser.add_argument('--device', type=str, default="1")
parser.add_argument('--no_context', default=False, action='store_true')
parser.add_argument('--neeg_dataset', default=False, action='store_true')
parser.add_argument('--story_cloze', default=False, action='store_true')
parser.add_argument('--candidates', type=int, default=5) # Narrative Cloze Task has 5 options
parser.add_argument('--conceptnet', default=False, action='store_true')
parser.add_argument('--semantic', default=False, action='store_true')
parser.add_argument('--input_size', type=int, default=10000)
args = parser.parse_args()

os.environ['TFHUB_CACHE_DIR'] = '.'
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import tensorflow_hub as hub
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K
import prepare_data
from prepare_data import tokenizer, tokenize_if_small_enough, pad_to_max_of_max, pad_to_max
import read
from new_model import build_model, initialize_vars
from keras.utils import to_categorical
# # Initialize session
sess = tf.Session()


### Read and split data
train_dataset = read.read_data_iterator(args.data)
features = list(tokenize_if_small_enough(train_dataset,
                                         args.sentence, args.no_context,
                                         is_neeg=args.neeg_dataset,
                                         conceptnet=args.conceptnet,
                                         input_size=args.input_size))
sample_size = len(features)
training_pct = 0.8
val_pct = 0.1
test_pct = 0.1
train_set_size = int(sample_size * training_pct)
val_set_size = int(sample_size * val_pct)
test_set_size = sample_size - train_set_size - val_set_size

train_features = features[:train_set_size]
val_features = features[train_set_size:train_set_size+val_set_size]
test_features = features[train_set_size+val_set_size:train_set_size+val_set_size+test_set_size]



#### Tokenize
input_ids = np.array([f.input_ids for f in features])
input_masks = np.array([f.input_mask for f in features])
segment_ids = np.array([f.segment_ids for f in features])
labels = np.array([to_categorical(f.label_id - 1, num_classes=5) for f in features])
word_events = pad_to_max_of_max([f.event_concept_vectors for f in features])
word_candidates = pad_to_max_of_max([f.candidate_concept_vectors for f in features])
sent_events = pad_to_max([[np.concatenate([f.event_sentence_pos[i], f.event_sentence_dep[i]]) 
                      for i in range(len(f.event_sentence_pos))] 
                     for f in features])
sent_candidates = np.zeros((sent_events.shape[0], 5, 1, sent_events.shape[-1]))

train_input_ids = input_ids[:train_set_size]
val_input_ids = input_ids[train_set_size:train_set_size+val_set_size]
train_input_masks = input_masks[:train_set_size]
val_input_masks = input_masks[train_set_size:train_set_size+val_set_size]
train_segment_ids = segment_ids[:train_set_size]
val_segment_ids = segment_ids[train_set_size:train_set_size+val_set_size]
train_labels = labels[:train_set_size]
val_labels = labels[train_set_size:train_set_size+val_set_size]
train_word_events = word_events[:train_set_size]
val_word_events = word_events[train_set_size:train_set_size+val_set_size]
train_word_candidates = word_candidates[:train_set_size]
val_word_candidates = word_candidates[train_set_size:train_set_size+val_set_size]
train_sent_events = sent_events[:train_set_size]
val_sent_events = sent_events[train_set_size:train_set_size+val_set_size]
train_sent_candidates = sent_candidates[:train_set_size]
val_sent_candidates = sent_candidates[train_set_size:train_set_size+val_set_size]




model = build_model(prepare_data.MAX_SEQ_LENGTH, num_labels=5,
                    word_vec_len=300 if args.conceptnet else 0, 
                    sentence_vec_len=169 if args.semantic else 0,
                    num_sents=train_word_events.shape[1])


initialize_vars(sess)


if args.mode == 'train':
    print(train_word_events.shape, val_word_events.shape)
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, '{epoch:02d}.h5')),
        keras.callbacks.TensorBoard(log_dir=args.output_dir, update_freq=1000)
    ]

    train_input = [train_input_ids, train_input_masks, train_segment_ids]
    val_input = [val_input_ids, val_input_masks, val_segment_ids]
    if args.conceptnet:
        train_input.extend([train_word_candidates, train_word_events])
        val_input.extend([val_word_candidates, val_word_events])
    if args.semantic:
        train_input.extend([train_sent_candidates, train_sent_events])
        val_input.extend([val_sent_candidates, val_sent_events])
    model.fit(
        train_input,
        train_labels,
        validation_data=(val_input, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

elif args.mode == 'predict':
    pass
