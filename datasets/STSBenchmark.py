import tensorflow as tf
from transformers import DataProcessor, InputExample
from transformers import InputFeatures, PreTrainedTokenizer
import pandas as pd
import os

class STSBenchmarkDataset():

  def __init__(self, tokenizer, max_length, **kwargs):
    self.tokenizer = tokenizer
    self.max_length = max_length

  def from_file(self, file_path):
    df = pd.read_csv(file_path).dropna()
    return self.from_dataframe(df)

  def from_dataframe(self, df, training=True):
    df["s1"] = df["s1"].str.lower()
    df["s2"] = df["s2"].str.lower()
    df["idx"] = df.index
    ds_tmp = tf.data.Dataset.from_tensor_slices(dict(df))
    return self._create_data_set(ds_tmp, training)

  def _convert_examples_to_features(
      self,
      examples,
  ):
    labels = [float(example.label) for example in examples]

    batch_encoding = self.tokenizer.batch_encode_plus(
      [(example.text_a, example.text_b) for example in examples], max_length=self.max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
      inputs = {k: batch_encoding[k][i] for k in batch_encoding}

      feature = InputFeatures(**inputs, label=labels[i])
      features.append(feature)

    return features

  def _create_data_set(self, examples, training=True):
    processor = CustomStsbProcessor()
    examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
    features = self._convert_examples_to_features(examples)

    def gen():
      for ex in features:
        if training:
          yield (
            {
              "input_ids": ex.input_ids,
              "attention_mask": ex.attention_mask,
            },
            ex.label,
          )
        else:
          yield (
            {
              "input_ids": ex.input_ids,
              "attention_mask": ex.attention_mask,
            }
          )

    if training:
      return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.float32),
        (
          {
            "input_ids": tf.TensorShape([None]),
            "attention_mask": tf.TensorShape([None]),
          },
          tf.TensorShape([]),
        ),
      )
    else:
      return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}),
        (
          {
            "input_ids": tf.TensorShape([None]),
            "attention_mask": tf.TensorShape([None]),
          }
        ),
      )


class CustomStsbProcessor(DataProcessor):

  def get_example_from_tensor_dict(self, tensor_dict):
    return InputExample(
      tensor_dict["idx"].numpy(),
      tensor_dict["s1"].numpy().decode("utf-8"),
      tensor_dict["s2"].numpy().decode("utf-8"),
      str(tensor_dict["score"].numpy()/5.0),
    )

  def get_labels(self):
    """See base class."""
    return [None]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[7]
      text_b = line[8]
      label = line[-1]
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

