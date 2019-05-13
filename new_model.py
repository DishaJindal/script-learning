import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import optimization
from bert import tokenization
from tensorflow import keras
import re
import prepare_data as prepare_data
from prepare_data import tokenize_if_small_enough


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            prepare_data.BERT_MODEL_HUB,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
            
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
    
class MultiBertLayer(BertLayer):
    def call(self, inputs):
        return [super(MultiBertLayer, self).call(ip) for ip in zip(*inputs)]

    def compute_output_shape(self, input_shape):
        #(batch size, num_labels, max_seq_size)
        return (input_shape[0], input_shape[1], self.output_size)
    
# Build model
def build_model(max_seq_length, num_labels, word_vec_len=0, sentence_vec_len=0, num_sents=5): 
    
    #Inputs
    in_ids = layers.Input(shape=(num_labels, max_seq_length), name="input_ids")
    in_masks = layers.Input(shape=(num_labels, max_seq_length), name="input_masks")
    in_segments = layers.Input(shape=(num_labels, max_seq_length), name="segment_ids")
    inputs = [in_ids, in_masks, in_segments]
    
    if word_vec_len:
        in_candidates_words = layers.Input(shape=(num_labels, None, word_vec_len), name="input_candidates_words")
        in_events_words = layers.Input(shape=(num_sents, None, word_vec_len), name="input_event_words")
        inputs.extend([in_candidates_words, in_events_words])
        
    if sentence_vec_len:
        in_candidates_sentences = layers.Input(shape=(num_labels, 1, sentence_vec_len), name="input_candidates_sentences")
        in_events_sentences = layers.Input(shape=(num_sents, sentence_vec_len), name="input_events_sentences")
        inputs.extend([in_candidates_sentences, in_events_sentences])
    
    
    #Split inputs if they should be operated on individually
    split_in_ids = [layers.Lambda(lambda x: x[:, i, :])(in_ids) for i in range(num_labels)]
    split_in_masks = [layers.Lambda(lambda x: x[:, i, :])(in_masks) for i in range(num_labels)]
    split_in_segments = [layers.Lambda(lambda x: x[:, i, :])(in_segments) for i in range(num_labels)]
    
    if word_vec_len:
        #Split by candidate
        split_in_candidates_words = [layers.Lambda(lambda x: x[:, i, :, :])(in_candidates_words) for i in range(num_labels)]
        #Split by sentence
        split_in_events_words = [layers.Lambda(lambda x: x[:, i, :, :])(in_events_words) for i in range(num_sents)]
        
    if sentence_vec_len:
        split_in_candidates_sentences = [layers.Lambda(lambda x: x[:, i, :])(in_candidates_sentences) 
                                         for i in range(num_labels)]
    
    
    #Bert
    bert_outputs = MultiBertLayer(n_fine_tune_layers=0)([split_in_ids, split_in_masks, split_in_segments])
    
    
    #Autoencoders to convert word and sentence embeddings into fixed vector embeddings
    
    
    word_to_sentence_autoencoder = layers.LSTM(sentence_vec_len if sentence_vec_len else 300,
                                               name="word_to_sword_to_sentence_autoencoder")
    sentences_to_vec_autoencoder = layers.LSTM(300, name='sentences_to_vec_autoencoder')
    
    if word_vec_len:
        autoencoded_word_event_sentence_vectors = [
            layers.Lambda(lambda x: K.expand_dims(word_to_sentence_autoencoder(x), 1))(ew) 
            for ew in split_in_events_words]
#         autoencoded_candidates = [
#             word_to_sentence_autoencoder(layers.Lambda(lambda x: K.expand_dims(x, 1))(cw))
#             for cw in split_in_candidates_words]
    
        autoencoded_candidates = [
            layers.Lambda(lambda x: K.expand_dims(x, 1))(word_to_sentence_autoencoder(cw))
            for cw in split_in_candidates_words
        ]
        word_event_candidate_autoencoded = [
            layers.Concatenate(axis=1)(autoencoded_word_event_sentence_vectors + [ac]) 
            for ac in autoencoded_candidates]

        autencoded_chains_from_words = [sentences_to_vec_autoencoder(wec) 
                                        for wec in word_event_candidate_autoencoded]
    if sentence_vec_len:
        sentence_candidate_vectors = [
            layers.Concatenate(axis=1)([in_events_sentences, ics]) 
            for ics in split_in_candidates_sentences]
        autencoded_chains_from_sents = [sentences_to_vec_autoencoder(scv) 
                                        for scv in sentence_candidate_vectors]
    
    if word_vec_len and sentence_vec_len:
        enhancing_vectors = [layers.Concatenate(axis=1)([wv, sv])
            for (wv, sv) in 
            zip(autencoded_chains_from_words, 
                autencoded_chains_from_sents)]
    elif word_vec_len:
        enhancing_vectors = autencoded_chains_from_words
    elif sentence_vec_len:
        enhancing_vectors = autencoded_chains_from_sents
        
        
    #Combine Bert and autoencoder embeddings (if provided)
    if word_vec_len or sentence_vec_len:
        augmented_outputs = [layers.Concatenate(axis=1)([bo, ev]) for (bo, ev) in zip(bert_outputs, enhancing_vectors)]
    else:
        augmented_outputs = bert_outputs
    concat_output = layers.Concatenate(axis=1)(augmented_outputs)
    
    #Single Hidden Layer and classification
    dense = layers.Dense(256, activation='relu')(concat_output)
    pred = layers.Dense(num_labels, activation='softmax')(dense)
    
    model = models.Model(inputs=inputs, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    print(inputs)
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
    
