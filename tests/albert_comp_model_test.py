import unittest
import tensorflow as tf
from models.AlbertForComparison import AlbertForComparison, get_albert_for_comparison
from datasets.STSBenchmark import STSBenchmarkDataset
import pandas as pd
from transformers import AlbertTokenizer

class TestAlbertForComparison(unittest.TestCase):

  def test_init_model(self):
    model = get_albert_for_comparison()
    print(model.summary())


  def test_load_weights(self):
    model = get_albert_for_comparison()
    model.load_weights('./.trained_models/albert_weights/ab.ckpt')

    df_test2 = pd.read_csv("../fakenews/data_for_validation/STSbenchmark/data/sts-test-cleaned.csv").dropna()
    df_test_in = df_test2.iloc[100:110]
    print(df_test_in)
    model_name = 'albert-base-v2'
    max_seq_length = 128

    tokenizer = AlbertTokenizer.from_pretrained(
      model_name,
      do_lower_case=True,
      add_special_tokens=True,
      max_length=max_seq_length,
      pad_to_max_length=True
    )
    pred_dataset = STSBenchmarkDataset(tokenizer, max_seq_length).from_dataframe(df_test_in, training=False)
    print(pred_dataset)
    result = model.predict(pred_dataset.batch(1)) * 5.0
    print(result)
    self.assertAlmostEqual(result[0], 0.058892, delta=0.00001)
    self.assertAlmostEqual(result[2], 4.0518007, delta=0.00001)
    self.assertAlmostEqual(result[-1], 3.647635, delta=0.00001)
    print(result)


if __name__ == '__main__':
  unittest.main()