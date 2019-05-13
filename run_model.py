import argparse
import os

parser = argparse.ArgumentParser(description='ScriptLearning')
parser.add_argument('mode', required=True, help="[train, predict]")
parser.add_argument('--data', type=str, default="dataset/gw_extractions_no_rep_no_fin.pickle")
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
from prepare_data import tokenizer, tokenize_if_small_enough
import read
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
train_input_ids = np.array([f.input_ids for f in train_features])
train_input_masks = np.array([f.input_mask for f in train_features])
train_segment_ids = np.array([f.segment_ids for f in train_features])
train_labels = np.array([to_categorical(f.label_id - 1, num_classes=5) for f in train_features])
train_word_events = [f.event_concept_vectors for f in train_features]
train_word_candidates = [f.candidate_concept_vectors for f in train_features]
train_sent_events = [np.concat([f.event_sentence_pos, f.event_sentence_dep]) for f in train_features]
train_sent_candidates = [np.zeros_like(train_sent_events[0]) for f in train_features]


val_input_ids = np.array([f.input_ids for f in val_features])
val_input_masks = np.array([f.input_mask for f in val_features])
val_segment_ids = np.array([f.segment_ids for f in val_features])
val_labels = np.array([to_categorical(f.label_id - 1, num_classes=5) for f in val_features])
val_word_events = [f.event_concept_vectors for f in val_features]
val_word_candidates = [f.candidate_concept_vectors for f in val_features]
val_sent_events = [np.concat([f.event_sentence_pos, f.event_sentence_dep]) for f in val_features]
val_sent_candidates = [np.zeros_like(train_sent_events[0]) for f in val_features]


model = build_model(prepare_data.MAX_SEQ_LENGTH, num_labels=5, word_vec_len=300, sentence_vec_len=0)

initialize_vars(sess)


if args.mode == 'train':
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(logs, '{epoch:02d}.h5')),
        keras.callbacks.TensorBoard(log_dir=logs, update_freq=1000)
    ]

    train_input = [train_input_ids, train_input_masks, train_segment_ids]
    val_input = [val_input_ids, val_input_masks, val_segment_ids]
    if args.conceptnet:
        train_input.extend([train_word_events, train_word_candidates])
        val_input.extend([val_word_events, val_word_candidates])
    if args.semantic:
        train_input.extend([train_sent_events, train_sent_candidates])
        val_input.extend([val_sent_events, val_sent_candidates])
    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids, train_aug_vecs], 
        train_labels,
        validation_data=([val_input_ids, val_input_masks, val_segment_ids, val_aug_vecs], val_labels),
        epochs=3,
        batch_size=1,
        callbacks=callbacks
    )

elif args.mode == 'predict':
    pass
