
import tensorflow as tf
from shapeChecker import ShapeChecker


# alternative would be Luong's multiplicative attention, to get this, change AdditiveAttention to Attention layers
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


# test the attention layer:
if __name__ == '__main__':
    samples = [
        "some sample text.",
        "another bit of sample text",
        "the story of texts",
    ]
    from machine_translator import create_text_processor
    from encoder import Encoder
    import matplotlib.pyplot as plt

    embedding_dim = 3
    units = 12
    input_text_processor = create_text_processor(samples, max_vocab_size=5)
    example_input_batch = tf.constant(samples, )
    example_tokens = input_text_processor(example_input_batch)
    # encode the tokens:
    encoder = Encoder(input_text_processor.vocabulary_size(),
                      embedding_dim, units)
    example_enc_output, example_enc_state = encoder(example_tokens)

    attention_layer = BahdanauAttention(units)
    #print((example_tokens != 0).shape)

    # Later, the decoder will generate this attention query
    example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])

    # Attend to the encoded tokens
    
    context_vector, attention_weights = attention_layer(
        query=example_attention_query,
        value=example_enc_output,
        mask=(example_tokens != 0))
    
    print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
    print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(attention_weights[:, 0, :])
    plt.title('Attention weights')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(example_tokens != 0)
    plt.title('Mask')
    
    print(attention_weights.shape)
    attention_slice = attention_weights[0, 0].numpy()
    attention_slice = attention_slice[attention_slice != 0]
    
    #@title
    plt.suptitle('Attention weights for one sequence')

    plt.figure(figsize=(12, 6))
    a1 = plt.subplot(1, 2, 1)
    plt.bar(range(len(attention_slice)), attention_slice)
    # freeze the xlim
    plt.xlim(plt.xlim())
    plt.xlabel('Attention weights')
    
    a2 = plt.subplot(1, 2, 2)
    plt.bar(range(len(attention_slice)), attention_slice)
    plt.xlabel('Attention weights, zoomed')
    
    # zoom in
    top = max(a1.get_ylim())
    zoom = 0.85*top
    a2.set_ylim([0.90*top, top])
    a1.plot(a1.get_xlim(), [zoom, zoom], color='k')
    plt.show()
