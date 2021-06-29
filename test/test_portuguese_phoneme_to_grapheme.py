
import unittest
import os


class TestPortuguesePhonemeToGrapheme(unittest.TestCase):

    def setUp(self) -> None:
        from src.disfluency_generator.portuguese_phoneme_to_grapheme import PhonemeToGrapheme
        self.repoRoot = os.path.abspath(os.path.join(os.getcwd(), '..'))

        test_sampa_tsv = os.path.join(self.repoRoot, "test", "resources", "test_sampa.tsv")
        self.p2g = PhonemeToGrapheme(test_sampa_tsv)

    def test_phoneme_list(self):
        # should be sorted longest (in terms of no. chars) to shortest
        expected_phonemes = ['6~j~', '6~', 'aw', 'p', 'Z']

        self.assertListEqual(self.p2g._sorted_phoneme_list, expected_phonemes)

    def test_phoneme_dict(self):
        expected_dict = {'p': 'p', 'Z': 'j', '6~': 'ã', 'aw': 'aw', '6~j~': 'êm'}

        self.assertDictEqual(self.p2g._phoneme2grapheme, expected_dict)

    def test_phoneme_to_grapheme(self):
        # not great tests long term because of the file dependency but for now good enough
        full_sampa_tsv = os.path.join(self.repoRoot, "resources", "sampa.tsv")
        from src.disfluency_generator.portuguese_phoneme_to_grapheme import PhonemeToGrapheme
        p2g = PhonemeToGrapheme(full_sampa_tsv)

        phoneme_string = "[\"viJu]"  # ['v', 'i', 'J', 'u']
        written_form = p2g.baseline_p2g(phoneme_string)
        expected_written_form = 'vinho'
        self.assertEqual(written_form, expected_written_form)

        phoneme_string = "[\"Ont6~j~]"  # ['O', 'n', 't', '6~j~']
        written_form = p2g.baseline_p2g(phoneme_string)
        expected_written_form = "ontêm"
        self.assertEqual(written_form, expected_written_form)

        p2g.sampa_df["p2g"] = p2g.sampa_df["Transcription"].apply(
            lambda x: p2g.baseline_p2g(x) if isinstance(x, str) else "")

        # now count the no. of Words == p2g to check we're mostly correct:
        df = p2g.sampa_df[p2g.sampa_df["Word"].notna()]
        matches = df[df["Word"] == df["p2g"]]
        misses = df[df["Word"] != df["p2g"]]

        # arbitarily require 80% full word accuracy:
        self.assertGreater(len(matches)/len(df), 0.80)

        # to set what might be improved:
        #print(misses)


if __name__ == '__main__':
    unittest.main()
