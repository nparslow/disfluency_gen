
import unittest


class TestDataPreparation(unittest.TestCase):
    def test_tf_lower_and_split_punct(self):
        from src.disfluency_generator.data_preparation import tf_lower_and_split_punct
        import tensorflow

        text = "A mãe do Flávio era florista."
        prepped_text_tensor = tf_lower_and_split_punct(text)
        self.assertIsInstance(prepped_text_tensor, tensorflow.Tensor)

        prepped_text = prepped_text_tensor.numpy().decode()
        # check that START and END are added,
        # lower case, punctuation removed
        # also (currently) accents removed
        expected_text = "[START] a mae do flavio era florista [END]"
        self.assertEqual(prepped_text, expected_text)


if __name__ == '__main__':
    unittest.main()
