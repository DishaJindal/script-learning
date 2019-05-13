import argparse
import os

parser = argparse.ArgumentParser(description='ScriptLearning')
parser.add_argument('--data', type=str, default="dataset/gw_extractions_no_rep_no_fin.pickle")  # 1
parser.add_argument('--device', type=str, default="1")

parser.add_argument('--sentence', default=False, action='store_true')
parser.add_argument('--no_context', default=False, action='store_true')
parser.add_argument('--neeg_dataset', default=False, action='store_true')
parser.add_argument('--story_cloze', default=False, action='store_true')
parser.add_argument('--conceptnet', default=False, action='store_true')

parser.add_argument('--candidates', type=int, default=5)  # Narrative Cloze Task has 5 options
parser.add_argument('--model_dir', type=str)
args = parser.parse_args()

os.environ['TFHUB_CACHE_DIR'] = '.'
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import read
import prepare_data
import input_builder
import importlib
import numpy as np
import model
import tensorflow as tf

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow_hub as hub
from datetime import datetime
from pdb import set_trace
import bert
from bert import optimization
from bert import tokenization
from tensorflow import keras
import re
from model import *
import sys
import prepare_data as prepare_data
from prepare_data import tokenize_if_small_enough
from read import *

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128

importlib.reload(input_builder)

model_dir = 'output_sentence'
run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_summary_steps=0,
    save_checkpoints_steps=0,
    log_step_count_steps=100)

model_fn = model.model_fn_builder(
    num_labels=5,
    learning_rate=LEARNING_RATE,
    num_train_steps=1,
    num_warmup_steps=1)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})

dataset = list(read.read_data_iterator(args.data))
val_data = dataset[int(0.8 * len(dataset)):]
print("Prediction Set Size: {}", len(val_data))

check_dataset = []
for ec_dict in val_data:
    check_dataset.append(ec_dict)

predict_set = list(
    prepare_data.tokenize_if_small_enough(check_dataset, sentences=args.sentence, no_context=args.no_context, is_neeg=args.neeg_dataset,
                                         conceptnet=args.conceptnet,
                                          input_size=len(val_data)))
predict_input_fn = input_builder.input_fn_builder(
    features=predict_set,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False,
    candidates=args.candidates)

predictions = estimator.predict(input_fn=predict_input_fn, yield_single_examples=False)
predictions_list = list(predictions)
bad = []
for i, c in enumerate(check_dataset):
    r = list(prepare_data.tokenize_if_small_enough([c], sentences=args.sentence, no_context=args.no_context))
    if not r:
        bad.append(i)

good_gt = [c for i, c in enumerate(check_dataset) if i not in bad]
pred_df = pd.DataFrame.from_dict({'predictions': predictions_list, 'dataset': good_gt})
print(len(good_gt))
print(len(predictions_list))
pred_df['pred_label'] = pred_df.predictions.apply(lambda x: x['labels'])
pred_df['gt_label'] = pred_df.dataset.apply(lambda x: x['correct'])
pred_df['correct_pred'] = pred_df.apply(lambda s: s.pred_label == s.gt_label, axis=1)
print('Accuracy: ', pred_df.correct_pred.sum() / len(pred_df.index))
