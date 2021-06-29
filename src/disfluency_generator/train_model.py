

import tensorflow as tf
from data_preparation import load_data, create_dataset, print_examples, tf_lower_and_split_punct,\
    create_text_processor
from encoder import Encoder
from decoder import Decoder
from trainTranslator import TrainTranslator, BatchLogs
from maskedLoss import MaskedLoss

from translator import Translator
from trainTranslator import TrainTranslator

import pathlib


def prepare_data(data_path, verbose=0):
    # 2 cols, eng\tspa, seems to be sorted by length
    data_filename = pathlib.Path(data_path, "spa-eng", "spa.txt")

    targets, inputs = load_data(data_filename)

    if verbose > 0:
        print(f"Last example of data:\n{inputs[-1]}\n{targets[-1]}")

    dataset = create_dataset(inputs, targets, BATCH_SIZE=64)

    if verbose > 0:
        print_examples(dataset)

    return inputs, targets, dataset



def create_text_processors(inputs, targets, dataset, verbose):

    # todo - check with corpus:
    max_vocab_size = 5000
    input_text_processor = create_text_processor(inputs, max_vocab_size)

    if verbose > 0:
        # todo better checking:
        print("First 10 words of input vocab:")
        print(input_text_processor.get_vocabulary()[:10])

    # note - we don't have to have the same output vocab size:
    output_text_processor = create_text_processor(targets, max_vocab_size)

    if verbose > 0:
        print("First 10 words of target vocab:")
        print(output_text_processor.get_vocabulary()[:10])

    if verbose > 0:
        for example_input_batch, example_target_batch in dataset.take(1):
            print("Example input token sequences (indices):")
            example_tokens = input_text_processor(example_input_batch)
            print(example_tokens[:3, :10])

    # todo make these class attributes
    return dataset, input_text_processor, output_text_processor


def train_model(dataset, input_text_processor, output_text_processor):
    embedding_dim = 256
    units = 1024

    train_translator = TrainTranslator(
        embedding_dim, units,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
        use_tf_function=True)  # False for slow

    # Configure the loss and optimizer
    train_translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(),
    )

    batch_loss = BatchLogs('batch_loss')

    train_translator.fit(dataset, epochs=3,
                         callbacks=[batch_loss])
    
    # monitoring
    import matplotlib.pyplot as plt
    plt.plot(batch_loss.logs)
    plt.ylim([0, 3])
    plt.xlabel('Batch #')
    plt.ylabel('CE/token')

    translator = Translator(
        encoder=train_translator.encoder,
        decoder=train_translator.decoder,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )

    # save - todo model name as variable
    tf.saved_model.save(translator, 'translator',
                        signatures={'serving_default': translator.tf_translate})


def reload():
    return tf.saved_model.load('translator')


if __name__ == '__main__':
    # todo input var:
    data_path = pathlib.Path("/home/nickp/Documents/tutorials/disfluency_generator/data")
    prepare_data(data_path, verbose=1)
