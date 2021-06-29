
import unittest
import os
import sys
from unittest import mock


class TestLetsReadPrepareTranslations(unittest.TestCase):
    def setUp(self) -> None:
        repoRoot = os.path.abspath(os.path.join(os.getcwd()))
        self.test_corpus_path = os.path.join(repoRoot, "test", "resources", "mini_LetsReadDB")

    def test_extract_annotation_no_p2g(self):
        from src.disfluency_generator.letsread_prepare_translations import LetsReadDataPrep

        lrdp = LetsReadDataPrep(self.test_corpus_path, p2g=None)
        sample_trs = os.path.join(self.test_corpus_path, "TRS", "c018_17_3_cor.trs")

        annotation = lrdp.extract_annotation(sample_trs)

        # note NOI, SIL will be dropped
        expected_annotation = "a solidão das suas crateras o deserto das suas planícies e a luz pálida das" \
                              " [S\"u6S] [viZ\"i...li6S] noturnas"

        self.assertEqual(annotation, expected_annotation)

        # todo test file not found

    @mock.patch('src.disfluency_generator.letsread_prepare_translations.PhonemeToGrapheme')
    def test_extract_annotation_with_p2g(self, mocked_p2g):
        mocked_p2g.baseline_p2g.side_effect = ["some_written_form_1", "some_written_form_2"]

        from src.disfluency_generator.letsread_prepare_translations import LetsReadDataPrep
        lrdp = LetsReadDataPrep(self.test_corpus_path, p2g=mocked_p2g)

        sample_trs = os.path.join(self.test_corpus_path, "TRS", "c018_17_3_cor.trs")

        annotation = lrdp.extract_annotation(sample_trs)

        # note NOI, SIL will be dropped
        expected_annotation = "a solidão das suas crateras o deserto das suas planícies e a luz pálida das" \
                              " some_written_form_1 some_written_form_2 noturnas"

        self.assertEqual(annotation, expected_annotation)


if __name__ == '__main__':
    unittest.main()