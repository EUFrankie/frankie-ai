import unittest
from frankie_ai.models.FrankieSentenceEncoder import FrankieSentenceEncoder
import numpy as np
import tensorflow as tf
from frankie_ai.datasets.STSBenchmark import STSBenchmarkDatasetForEncoding

class TestFrankieEncoder(unittest.TestCase):

  # def test_init_model(self):
  #   tf.get_logger().setLevel('ERROR')
    # encoder = FrankieSentenceEncoder()
    # print(encoder.get_encoder_model().summary())
    # #self.assertTrue("66,362,880" in str(encoder.get_encoder_model().summary()))

  def setUp(self) -> None:
    tf.get_logger().setLevel('ERROR')
    self.encoder = FrankieSentenceEncoder(DatasetClass=STSBenchmarkDatasetForEncoding)

  def test_senctences_encoding_larger_batch(self):
    sentences = [
      "Frankie is going to make publishing COVID-19 fake news pointless",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges."
    ]
    self.assertEqual(self.encoder.encode(sentences).shape, (9, 768))

  def test_senctences_encoding_smaller_batch(self):
    sentences = [
      "Frankie is going to make publishing COVID-19 fake news pointless",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges."
    ]
    self.assertEqual(self.encoder.encode(sentences).shape, (6, 768))

  def test_senctences_encoding_equals_batch(self):
    sentences = [
      "Frankie is going to make publishing COVID-19 fake news pointless",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
      "Working on Frankie helps solving one of the pandemic challenges.",
    ]
    self.assertEqual(self.encoder.encode(sentences).shape, (8, 768))

  # def tearDown(self) -> None:
  #   tf.get_logger().setLevel('INFO')


if __name__ == '__main__':
  unittest.main()