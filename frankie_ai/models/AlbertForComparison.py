import tensorflow as tf
from transformers import AlbertConfig, AlbertTokenizer, TFAlbertModel

def get_albert_for_comparison():
  model_name = 'albert-base-v2'
  config = AlbertConfig.from_pretrained(model_name)
  config.output_hidden_states = False

  input_ids = tf.keras.Input(shape=(128,), name='input_ids', dtype=tf.int32)
  attention_mask = tf.keras.Input(shape=(128,), name='attention_mask', dtype=tf.int32)

  transformer_model = TFAlbertModel.from_pretrained(model_name, config=config)
  embedding_layer = transformer_model([input_ids, attention_mask])[0]

  X = tf.keras.layers.Dense(
    config.hidden_size,
    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
    activation="relu",
    name="pre_classifier",
  )(embedding_layer[:, 0])
  X = tf.keras.layers.Dropout(config.classifier_dropout_prob)(X)
  output_ = tf.keras.layers.Dense(
    1, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name="classifier"
  )(X)

  return tf.keras.Model([input_ids, attention_mask], output_)


class AlbertForComparison(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super(AlbertForComparison, self).__init__(self, args, kwargs)
    self.model_name = 'albert-base-v2'
    self.config = AlbertConfig.from_pretrained(self.model_name)
    self.config.output_hidden_states = False

    self.embedding_layer = TFAlbertModel.from_pretrained(self.model_name, config=self.config)
    self.pre_classifier = tf.keras.layers.Dense(
        self.config.hidden_size,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.initializer_range),
        activation="relu",
        name="pre_classifier",
    )
    self.classifier = tf.keras.layers.Dense(
      1,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.initializer_range),
      name="classifier"
    )

  def call(self, inputs, training=False):
    x = self.embedding_layer(inputs)[0] # Only select last hidden states output (0)
    x = self.pre_classifier(x[:, 0]) # Select CLS Embedding (pos 0) for classification
    if training:
      x = tf.keras.layers.Dropout(self.config.classifier_dropout_prob)(x) # Add dropout for improved generalization
    return self.classifier(x)