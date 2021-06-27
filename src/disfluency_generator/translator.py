
import tensorflow as tf
import numpy as np
from shapeChecker import ShapeChecker
from machine_translator import tf_lower_and_split_punct
from decoder import DecoderInput

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


class Translator(tf.Module):
    def __init__(self,
                 encoder, decoder,
                 input_text_processor,
                 output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                invert=True))

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary())
        token_mask_ids = index_from_string(['',
                                            '[UNK]',
                                            '[START]']).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string('[START]')
        self.end_token = index_from_string('[END]')

    def tokens_to_text(self, result_tokens):
        shape_checker = ShapeChecker()
        shape_checker(result_tokens, ('batch', 't'))
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        shape_checker(result_text_tokens, ('batch', 't'))

        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')
        shape_checker(result_text, ('batch'))

        result_text = tf.strings.strip(result_text)
        shape_checker(result_text, ('batch',))
        return result_text

    def sample(self, logits, temperature):
        shape_checker = ShapeChecker()
        # 't' is usually 1 here.
        shape_checker(logits, ('batch', 't', 'vocab'))
        shape_checker(self.token_mask, ('vocab',))

        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits / temperature,
                                               num_samples=1)

        shape_checker(new_tokens, ('batch', 't'))

        return new_tokens

    def translate_unrolled(self,
                           input_text, *,
                           max_length=50,
                           return_attention=True,
                           temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output,
                                     mask=(input_tokens != 0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        #return self.translate_unrolled(input_text)  # slower, todo - investigate more
        return self.translate_symbolic(input_text)

    # @title [Optional] Use a symbolic loop
    def translate_symbolic(self,
                           input_text, *,
                           max_length=50,
                           return_attention=True,
                           temperature=1.0):
        shape_checker = ShapeChecker()
        shape_checker(input_text, ('batch',))

        batch_size = tf.shape(input_text)[0]

        # Encode the input
        input_tokens = self.input_text_processor(input_text)
        shape_checker(input_tokens, ('batch', 's'))

        enc_output, enc_state = self.encoder(input_tokens)
        shape_checker(enc_output, ('batch', 's', 'enc_units'))
        shape_checker(enc_state, ('batch', 'enc_units'))

        # Initialize the decoder
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)
        shape_checker(new_tokens, ('batch', 't1'))

        # Initialize the accumulators
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
        attention = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        shape_checker(done, ('batch', 't1'))

        for t in tf.range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output,
                                     mask=(input_tokens != 0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
            attention = attention.write(t, dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)
            shape_checker(dec_result.logits, ('batch', 't1', 'vocab'))
            shape_checker(new_tokens, ('batch', 't1'))

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens = result_tokens.write(t, new_tokens)

            if tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = result_tokens.stack()
        shape_checker(result_tokens, ('t', 'batch', 't0'))
        result_tokens = tf.squeeze(result_tokens, -1)
        result_tokens = tf.transpose(result_tokens, [1, 0])
        shape_checker(result_tokens, ('batch', 't'))

        result_text = self.tokens_to_text(result_tokens)
        shape_checker(result_text, ('batch',))

        if return_attention:
            attention_stack = attention.stack()
            shape_checker(attention_stack, ('t', 'batch', 't1', 's'))

            attention_stack = tf.squeeze(attention_stack, 2)
            shape_checker(attention_stack, ('t', 'batch', 's'))

            attention_stack = tf.transpose(attention_stack, [1, 0, 2])
            shape_checker(attention_stack, ('batch', 't', 's'))

            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}


# @title Labeled attention plots
def plot_attention(attention, sentence, predicted_sentence):
    sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()
    predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    attention = attention[:len(predicted_sentence), :len(sentence)]

    ax.matshow(attention, cmap='viridis', vmin=0.0)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')
    plt.suptitle('Attention weights')


if __name__ == "__main__":
    example_output_tokens = tf.random.uniform(
        shape=[5, 2], minval=0, dtype=tf.int64,
        maxval=output_text_processor.vocabulary_size())
    translator.tokens_to_text(example_output_tokens).numpy()

    #% % time
    input_text = tf.constant([
        'hace mucho frio aqui.',  # "It's really cold here."
        'Esta es mi vida.',  # "This is my life.""
    ])

    result = translator.translate(
        input_text=input_text)

    print(result['text'][0].numpy().decode())
    print(result['text'][1].numpy().decode())
    print()

    #% % time
    result = translator.tf_translate(
        input_text=input_text)

    #% % time
    result = translator.tf_translate(
        input_text=input_text)

    print(result['text'][0].numpy().decode())
    print(result['text'][1].numpy().decode())
    print()

    #% % time
    result = translator.translate(
        input_text=input_text)

    print(result['text'][0].numpy().decode())
    print(result['text'][1].numpy().decode())
    print()


    a = result['attention'][0]
    print(np.sum(a, axis=-1))

    _ = plt.bar(range(len(a[0, :])), a[0, :])
    plt.imshow(np.array(a), vmin=0.0)

    # checking attention

    i = 0
    plot_attention(result['attention'][i], input_text[i], result['text'][i])
    print(result['text'])
    #% % time
    three_input_text = tf.constant([
        # This is my life.
        'Esta es mi vida.',
        # Are they still home?
        '¿Todavía están en casa?',
        # Try to find out.'
        'Tratar de descubrir.',
    ])

    result = translator.tf_translate(three_input_text)

    for tr in result['text']:
        print(tr.numpy().decode())

    print()

    i = 0
    plot_attention(result['attention'][i], three_input_text[i], result['text'][i])
    i = 1
    plot_attention(result['attention'][i], three_input_text[i], result['text'][i])
    i = 2
    plot_attention(result['attention'][i], three_input_text[i], result['text'][i])

    # checking longer texts
    long_input_text = tf.constant([inp[-1]])

    import textwrap

    print('Expected output:\n', '\n'.join(textwrap.wrap(targ[-1])))
    result = translator.tf_translate(long_input_text)

    i = 0
    plot_attention(result['attention'][i], long_input_text[i], result['text'][i])
    _ = plt.suptitle('This never works')