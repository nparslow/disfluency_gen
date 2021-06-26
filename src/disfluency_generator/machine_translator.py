
# from : https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/nmt_with_attention.ipynb#scrollTo=TqHsArVZ3jFS
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers.experimental import preprocessing


def load_data(path):
    # path = Path lib object (not string)
    text = path.read_text(encoding='utf-8')

    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]

    return targ, inp


def create_dataset(inputs, targets):
    BUFFER_SIZE = len(inputs)
    BATCH_SIZE = 64

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def print_examples(dataset):
    for example_input_batch, example_target_batch in dataset.take(1):
        print(example_input_batch[:5])
        print()
        print(example_target_batch[:5])
        break


def tf_lower_and_split_punct(text):
    # todo - the regexes below are language specific
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')  # NFKD re: http://unicode.org/reports/tr15/ like ICU
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def create_text_processor(samples, max_vocab_size):
    text_processor = preprocessing.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size)
    text_processor.adapt(samples)
    return text_processor
