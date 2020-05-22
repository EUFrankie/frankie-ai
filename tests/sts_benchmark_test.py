import unittest
from datasets.STSBenchmark import STSBenchmarkDataset, STSBenchmarkDatasetForEncoding
from transformers import AlbertTokenizer
import tensorflow as tf
from transformers import DistilBertTokenizer

class TestSTSBenchmark(unittest.TestCase):

  def test_init(self):
    model_name = 'albert-base-v2'
    max_seq_length = 128

    tokenizer = AlbertTokenizer.from_pretrained(
      model_name,
      do_lower_case=True,
      add_special_tokens=True,
      max_length=max_seq_length,
      pad_to_max_length=True
    )

    ds = STSBenchmarkDataset(tokenizer, max_seq_length).from_file("/home/jan/fakenews/data_for_validation/STSbenchmark/data/sts-train-cleaned.csv")
    self.assertEqual(str(ds), '<FlatMapDataset shapes: ({input_ids: (None,), attention_mask: (None,)}, ()), types: ({input_ids: tf.int32, attention_mask: tf.int32}, tf.float32)>')


  def test_sentence_list(self):
    model_name = 'distilbert-base-uncased'
    max_seq_length = 64

    tokenizer = DistilBertTokenizer.from_pretrained(
      model_name,
      do_lower_case=True,
      add_special_tokens=True,
      max_length=max_seq_length,
      pad_to_max_length=True
    )

    sentences = [
      "Frankie is going to make publishing COVID-19 fake news pointless",
      "Working on Frankie helps solving one of the pandemic challenges."
    ]

    sts_ds = STSBenchmarkDatasetForEncoding(tokenizer, max_seq_length).from_sentence_list(sentences)
    ds = sts_ds.take(2)
    out = next(iter(ds))
    out = out['input_ids'].numpy()
    self.assertEqual(out[0], 101)
    self.assertEqual(out[1], 12784)
    self.assertEqual(out[2], 2003)
    self.assertEqual(out[7], 2522)
    self.assertEqual(out[12], 2739)
    self.assertEqual(out[14], 102)
    self.assertEqual(out[15], 0)

    out = next(iter(ds))
    out = out['input_ids'].numpy()
    self.assertEqual(out[11], 8275)

if __name__ == '__main__':
  unittest.main()