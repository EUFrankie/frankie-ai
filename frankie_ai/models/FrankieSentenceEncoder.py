import tensorflow as tf
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertModel
import os, sys
import numpy as np


class PoolingLayer(tf.keras.layers.Layer):

  def __init__(self, pooling_type):
    super(PoolingLayer, self).__init__()
    self.pooling_type = pooling_type

  def call(self, inputs):
    if self.pooling_type == "mean":
      shape_input = tf.shape(inputs[0])
      tile_shape = tf.stack([1, 1, shape_input[2]], 0)
      expanded_mask = tf.expand_dims(inputs[1], 2)
      inp_mask = tf.tile(
        tf.cast(expanded_mask, tf.float32),
        tile_shape
      )

      masked_word_embeddings = tf.math.multiply(inputs[0], inp_mask)
      return tf.math.reduce_mean(masked_word_embeddings, axis=1)
    elif self.pooling_type == "cls":
      return inputs[0][:, 0]
    else:
      print("WARNING! Polling Strategy not implemented return None")
      return None


class FrankieSentenceEncoder:

  def __init__(self, DatasetClass, weights_path = './.trained_models/frankie_encoder/ec.ckpt'):

    # transformer/model parameters are hardcoded due to usage of pre-trained weights
    self.max_seq_length = 64
    self.model_name = 'distilbert-base-uncased'

    self.tokenizer = tokenizer = DistilBertTokenizer.from_pretrained(
        self.model_name,
        do_lower_case=True,
        add_special_tokens=True,
        max_length=self.max_seq_length,
        pad_to_max_length=True
    )
    self.dataset = DatasetClass(self.tokenizer, self.max_seq_length)

    self.model_config = DistilBertConfig.from_pretrained(self.model_name)
    self.model_config.output_hidden_states = False
    self.model = self._create_sentence_transformer(input_shape=(self.max_seq_length,))
    self.model.load_weights(weights_path)
    print("Initialized Encoder Model")

  def get_encoder_model(self):
    return self.model

  def _create_sentence_transformer(self, input_shape):
    input_ids = tf.keras.Input(shape=input_shape, name='input_ids', dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=input_shape, name='attention_mask', dtype=tf.int32)
    transformer_model = TFDistilBertModel.from_pretrained(self.model_name, config = self.model_config)
    word_embedding_layer = transformer_model([input_ids, attention_mask])[0]
    sentence_embedding_layer = PoolingLayer(pooling_type="mean")([word_embedding_layer, attention_mask])
    return tf.keras.Model([input_ids, attention_mask], sentence_embedding_layer)

  def encode(self, senteces):

    batch_size = 8
    ds_sentences = self.dataset.from_sentence_list(senteces)

    embeddings = []
    for step, x in enumerate(ds_sentences.batch(batch_size)):
      embedding_batch = self.model.predict([x['input_ids'], x['attention_mask']])
      embeddings.append(embedding_batch)

    return np.concatenate(embeddings, axis=0)
