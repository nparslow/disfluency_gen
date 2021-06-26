
from machine_translator import load_data, create_dataset, print_examples, tf_lower_and_split_punct,\
    create_text_processor

import pathlib


def main(data_path, verbose=0):
    # 2 cols, eng\tspa, seems to be sorted by length
    data_filename = pathlib.Path(data_path, "spa-eng", "spa.txt")

    targets, inputs = load_data(data_filename)

    if verbose > 0:
        print(f"Last example of data:\n{inputs[-1]}\n{targets[-1]}")

    dataset = create_dataset(inputs, targets)

    if verbose > 0:
        print_examples(dataset)

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
        example_input_batch, example_target_batch = dataset.take(1)[0]
        print("Example input tokens:")
        example_tokens = input_text_processor(example_input_batch)
        print(example_tokens[:3, :10])





if __name__ == '__main__':
    # todo input var:
    data_path = pathlib.Path("/home/nickp/Documents/tutorials/disfluency_generator/data")
    main(data_path, verbose=1)