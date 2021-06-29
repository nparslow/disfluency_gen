
import tensorflow as tf
from .shapeChecker import ShapeChecker


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                   embedding_dim)

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       # Return the sequence and state
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's'))

        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(tokens)
        shape_checker(vectors, ('batch', 's', 'embed_dim'))

        # 3. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(vectors, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch', 'enc_units'))

        # 4. Returns the new sequence and its state.
        return output, state


# testing the encoder:
if __name__ == '__main__':
    from data_preparation import create_text_processor

    samples = [
        "some sample text.",
        "another bit of sample text",
        "the story of texts",
    ]
    input_text_processor = create_text_processor(samples, max_vocab_size=5)
    example_input_batch = tf.constant(samples,)
    embedding_dim = 3
    units = 12

    # Convert the input text to tokens.
    example_tokens = input_text_processor(example_input_batch)

    # Encode the input sequence.
    encoder = Encoder(input_text_processor.vocabulary_size(),
                      embedding_dim, units)
    example_enc_output, example_enc_state = encoder(example_tokens)

    # tokens will have shape = max length + 2 (for start and stop) = 7 here
    print(f'Input batch, shape (batch): {example_input_batch.shape}')
    print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
    print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
    print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')
