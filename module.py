import numpy as np
import tensorflow as tf
from tensorflow import keras

class Seq2SeqEncoder(keras.Model):
    def __init__(self,vocab_size, embedding_dims, hidden_units):
        super(Seq2SeqEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.hidden_units = hidden_units
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dims)
        self.gru = keras.layers.GRU(self.hidden_units,
                                    return_sequences = True,
                                    return_state = True,
                                    recurrent_initializer='glorot_uniform')


    def call(self, x, hidden):
        # x : (N, time_steps)
        x = self.embedding(x) # x: (N, time_steps, embedding_dims)

        output, hidden_state = self.gru(x, initial_state = hidden)# output : (N, time_steps, hidden_units) state : (N, hidden_units)
        return output, hidden_state

    def initial_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.hidden_units))


class LuongAttention(keras.layers.Layer):
    def __init__(self):
        super(LuongAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape = [input_shape[-1], input_shape[-1]],
                                 name = 'weight-att',
                                 initializer = 'uniform',
                                 trainable = True)
        super(LuongAttention, self).build(input_shape)

    def call(self, encoding_output, state):
        temp = tf.matmul(state, self.W) # (N, 1, hidden_units)
        return tf.keras.backend.batch_dot(temp, tf.transpose(encoding_output, (0, 2, 1))) # (N, 1, time_steps)


class Seq2SeqAttention(keras.Model):
    def __init__(self, type, units=10):
        assert type == 'bahdanau' or type == 'luong' or type == 'google'
        super(Seq2SeqAttention, self).__init__()
        self.type = type
        if self.type == 'luong':
            self.kernel = LuongAttention()
        if self.type == 'bahdanau':
            self.W1 = keras.layers.Dense(units)
            self.W2 = keras.layers.Dense(units)
            self.V = keras.layers.Dense(1)

    def call(self, encoding_output, state):

        assert encoding_output.shape[0] == state.shape[0]
        assert encoding_output.shape[-1] == state.shape[-1]

        state = tf.expand_dims(state, axis = 1) # (N, 1, hidden_units)
        if self.type == 'google':
            score = keras.backend.batch_dot(state, tf.transpose(encoding_output, (0, 2, 1))) # (N, 1, time_steps)


        if self.type == 'luong':
            score = self.kernel(encoding_output, state) # (N, 1, time_steps)


        if self.type == 'bahdanau':
            score = self.V(self.W1(encoding_output) + self.W2(state)) # (N, time_steps , 1)
            score = tf.transpose(score, (0, 2, 1)) # (N, 1, time_steps)

        weights = tf.transpose(tf.nn.softmax(score, axis=-1), (0, 2, 1))
        return tf.reduce_mean(weights * encoding_output, axis=1), weights

class Seq2SeqDecoder(keras.Model):

    def __init__(self, vocab_size, embedding_dims, hidden_units, combined='concat', attention_type='google'):
        assert combined == 'concat' or (combined == 'add' and hidden_units == embedding_dims)
        super(Seq2SeqDecoder, self).__init__()
        self.combined = combined
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.hidden_units = hidden_units
        self.attention_type = attention_type
        self.attention = Seq2SeqAttention(self.attention_type)
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dims)
        self.gru = keras.layers.GRU(hidden_units,
                                    return_sequences = True,
                                    return_state = True,
                                    recurrent_initializer='glorot_uniform')
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, x, hidden_state, encoding_output):
        # x : (N, 1)
        # state: (N, hidden_units)
        # encoding_ouput: (N, time_steps, hidden_units)
        assert hidden_state.shape[-1] == encoding_output.shape[-1]
        context, weights = self.attention(encoding_output, hidden_state) # context : (N, hidden_units), weights: (N, time_steps, 1)
        x = self.embedding(x) # (N, 1, embedding_dims)
        if self.combined == 'concat':
            x = tf.concat([x, tf.expand_dims(context, axis=1)], axis = -1) # (N, 1, hidden_units + emdedding_dims)
        else:
            x = x + tf.expand_dims(context, axis=1) # (N, 1, hidden_units)

        output, state = self.gru(x, initial_state = hidden_state) # (N, 1, hidden_units) (N, hidden_units)

        output = tf.squeeze(output, axis = 1)

        output = self.dense(output)

        return output, state, weights
