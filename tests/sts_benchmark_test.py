import unittest
from datasets.STSBenchmark import STSBenchmarkDataset
from transformers import AlbertConfig, AlbertTokenizer, TFAlbertModel

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
    #self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
  unittest.main()