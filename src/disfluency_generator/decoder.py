
import tensorflow as tf
import typing
from typing import Any, Tuple
from .bahdanauAttention import BahdanauAttention
from .shapeChecker import ShapeChecker


class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                   embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    # todo faster implementation from:
    # https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/text_generation.ipynb
    def call(self,
             inputs: DecoderInput,
             state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        shape_checker(inputs.mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        # Step 1. Lookup the embeddings
        vectors = self.embedding(inputs.new_tokens)
        shape_checker(vectors, ('batch', 't', 'embedding_dim'))

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weights), state


if __name__ == '__main__':
    from data_preparation import create_text_processor
    from encoder import Encoder
    import numpy as np

    input_samples = [
        "some sample text.",
        "another bit of sample text",
        "the story of texts",
    ]
    target_samples = [
        "some sa sample text.",
        "another another bit of sample text",
        "the story of um texts",
    ]
    output_text_processor = create_text_processor(target_samples, max_vocab_size=7)
    embedding_dim = 3
    units = 12
    example_target_batch = tf.constant(target_samples, )

    input_text_processor = create_text_processor(input_samples, max_vocab_size=5)
    example_input_batch = tf.constant(input_samples, )
    example_tokens = input_text_processor(example_input_batch)
    # encode the tokens:
    input_text_processor = create_text_processor(input_samples, max_vocab_size=5)
    encoder = Encoder(input_text_processor.vocabulary_size(),
                      embedding_dim, units)
    example_enc_output, example_enc_state = encoder(example_tokens)

    # testing the decoder:
    decoder = Decoder(output_text_processor.vocabulary_size(),
                      embedding_dim, units)

    # Convert the target sequence, and collect the "[START]" tokens
    example_output_tokens = output_text_processor(example_target_batch)

    start_index = output_text_processor._index_lookup_layer('[START]').numpy()
    first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])

    # Run the decoder
    dec_result, dec_state = decoder(
        inputs=DecoderInput(new_tokens=first_token,
                            enc_output=example_enc_output,
                            mask=(example_tokens != 0)),
        state=example_enc_state
    )

    print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')
    print(f'state shape: (batch_size, dec_units) {dec_state.shape}')

    sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)

    vocab = np.array(output_text_processor.get_vocabulary())
    first_word = vocab[sampled_token.numpy()]
    print("first words:", first_word[:5])

    # second set of logits:
    dec_result, dec_state = decoder(
        DecoderInput(sampled_token,
                     example_enc_output,
                     mask=(example_tokens != 0)),
        state=dec_state)

    sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)
    first_word = vocab[sampled_token.numpy()]
    print("first words:", first_word[:5])