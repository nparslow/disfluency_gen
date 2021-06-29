
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re

from .portuguese_phoneme_to_grapheme import PhonemeToGrapheme


class LetsReadDataPrep:
    def __init__(self, corpus_path, p2g: PhonemeToGrapheme=None):
        self.corpus_path = corpus_path
        self.p2g = p2g
        self.df = None

    def extract_annotation(self, trs_filename, verbose=0) -> str:
        # note trsfile library fails - might be a python version clash
        # so using solution from:
        # https://stackoverflow.com/questions/61833003/how-to-parse-trs-xml-file-for-text-between-self-closing-tags
        try:
            tree = ET.parse(trs_filename)
        except FileNotFoundError:
            if verbose > 0:
                print(f"Cannot find file {trs_filename}, skipping")
            return np.nan
        root = tree.getroot()
        data = [text.strip() for node in root.findall('.//Turn') for text in node.itertext() if text.strip()]

        spoken_text = []
        for element in data:
            tag = re.search(r"^\*[A-Z]{3}", element)
            if tag:
                # todo - should be able to just skip the * with a () group
                tag = tag.group().strip("*")
                if tag not in ("NOI", "SIL"):
                    params = re.search(r"\([^)]+\)", element)
                    if params:
                        spoken = params.group().strip("()")
                        if ',' in spoken:
                            prompt, spoken = spoken.split(',')
                        spoken = spoken.strip()

                        if re.match(r'^\[[^\]]+\]$', spoken) and self.p2g is not None:
                            approximate_written_form = self.p2g.baseline_p2g(spoken)
                            spoken = approximate_written_form
                        elif ("[" in spoken or "]" in spoken) and self.p2g is not None:
                            print(f"Likely typo: {spoken}")

                        spoken_text.append(spoken)
            else:
                spoken_text.append(element)

        return " ".join(spoken_text)

    def prep_letsread(self):
        main_tsv = os.path.join(self.corpus_path, "LetsReadDB_main.txt")
        self.df = pd.read_csv(main_tsv, sep='\t', encoding="utf-8")

        extract_spoken = lambda audio_id: self.extract_annotation(
            os.path.join(self.corpus_path, "TRS", f"{audio_id}_cor.trs")
        )
        self.df["Spoken"] = self.df["AudioID"].apply(extract_spoken)

        filtered_df = self.df[self.df["Spoken"].notna()]
        return list(filtered_df["Prompt"][filtered_df.Spoken.str.len() > 0]),\
               list(filtered_df["Spoken"][filtered_df.Spoken.str.len() > 0])

    def vocab_histogram(self):
        filtered_df = self.df[self.df["Spoken"].notna()]
        # outputs
        all_words = filtered_df.Spoken.str.split(expand=True).stack()
        # 25584 tokens with
        # 5138 unique tokens
        # 2915 with more than 1 appearance
        word_counts = all_words.value_counts().reset_index(name='counts')
        print(word_counts[word_counts["counts"]>1])
        #print(all_words.value_counts())
        #import matplotlib.pyplot as plt
        #word_counts.plot(kind='bar')
        #plt.show()


def main():

    repoRoot = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    corpus_path = os.path.join(repoRoot, "data", "LetsReadDB")

    p2g = PhonemeToGrapheme(os.path.join(repoRoot, "resources", "sampa.tsv"))
    lrdp = LetsReadDataPrep(corpus_path, p2g=p2g)
    inputs, targets = lrdp.prep_letsread()
    lrdp.vocab_histogram()


    # extract_annotation("a001_01_2")

    # testing:
    """
    for i, row in df.iterrows():
        prompt = row["Prompt"]
        audio_id = row["AudioID"]

        trs_filename = get_trs_filename(corpus_path, audio_id)
        print(extract_annotation(trs_filename))

        if i > 10:
            break

    print(df.head())
    print(df.shape)
    print(df["Spoken"].isna().sum())
    """


if __name__ == '__main__':
    main()
