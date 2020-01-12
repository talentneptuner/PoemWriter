import tensorflow as tf
from datetime import datetime
from tensorflow import keras

from preprecessing import *
from hyperParameters import HyperParmaters
from module import *

def loss_function(labels, logits, loss_object):
    mask = 1.0 - tf.cast(tf.math.equal(labels, 0), tf.float64)

    loss_ = loss_object(labels, logits)
    mask = tf.cast(mask, loss_.dtype)

    loss_ *= mask
    return  tf.reduce_mean(loss_)


def train_step(x, y, initial_hidden, encoder, decoder, loss_object, optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        encoder_output, encoder_hidden = encoder(x, initial_hidden)
        decoder_hidden = encoder_hidden
        for t in range(y.shape[-1] - 1):
            decoder_input = tf.expand_dims(y[:, t], axis=1)
            decoder_output, decoder_hidden, weights = decoder(decoder_input, decoder_hidden, encoder_output)
            loss_step = loss_function(y[:, t + 1], decoder_output, loss_object)
            loss += loss_step
    total_loss  = loss / y.shape[0]

    parmeters = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, parmeters)
    # print(gradients, parmeters)
    optimizer.apply_gradients(zip(gradients, parmeters)) # apply_gradients 需要的是(梯度， 参数)的集合
    return total_loss


def evaluate_training(sentence, tokenizer, encoder, decoder, hidden_units):
    inputs= tokenizer.se_to_idx(sentence)
    inputs = [tokenizer.atom['<bos>']] + inputs + [tokenizer.atom['<eos>']]
    attention_weights = np.zeros((len(inputs), len(inputs)))
    inputs = tf.convert_to_tensor(inputs)

    results = ''

    initial_hidden = tf.zeros((1, hidden_units))
    inputs = tf.expand_dims(inputs, axis=0)

    encoder_output, encoder_hidden = encoder(inputs, initial_hidden)

    decoder_hidden = encoder_hidden

    decoder_input = tf.expand_dims([tokenizer.atom['<bos>']], 0)

    for t in range(attention_weights.shape[0] - 1):
        predictions, decoder_hidden, weights = decoder(
            decoder_input, decoder_hidden,
            encoder_output)  # attention_weights : (1, input_length, 1) predictions: (1, vocab_size)
        attention_weights[:, t] = tf.reshape(weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        results += tokenizer.vocab[predicted_id]

        decoder_input = tf.expand_dims([predicted_id], 0)

    return results[:-1], sentence, attention_weights



def train(hyperparmeters:HyperParmaters):
    tokenizer = Tokenizer(hyperparmeters.tokenizer_type)
    tokenizer.fit(hyperparmeters.tokenizer_file)
    if hyperparmeters.save_tokenizer:
        tokenizer.save_to_json(hyperparmeters.tokenizer_save_path)
    dataset, num_examples = get_dataset(hyperparmeters.train_filepath, tokenizer,
                          hyperparmeters.batch_size)

    encoder = Seq2SeqEncoder(len(tokenizer.vocab),
                             hyperparmeters.embedding_dims,
                             hyperparmeters.hidden_units)
    hidden_state = encoder.initial_hidden_state(hyperparmeters.batch_size)
    decoder = Seq2SeqDecoder(len(tokenizer.vocab),
                             hyperparmeters.embedding_dims,
                             hyperparmeters.hidden_units,
                             attention_type = hyperparmeters.attention_type,
                             combined='concat')
    optimizer = keras.optimizers.Adam()
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = True,
                                                             reduction = 'none')
    steps_per_epoch = num_examples // hyperparmeters.batch_size

    if hyperparmeters.model_save_path:
        if not os.path.exists(hyperparmeters.model_save_path):
            os.mkdir(hyperparmeters.model_save_path)
            os.mkdir(os.path.join(hyperparmeters.model_save_path, 'encoder'))
            os.mkdir(os.path.join(hyperparmeters.model_save_path, 'decoder'))
        encoder_path = os.path.join(hyperparmeters.model_save_path, 'encoder')
        decoder_path = os.path.join(hyperparmeters.model_save_path, 'decoder')

    sample = '两个黄鹂鸣翠柳'
    for epoch in range(hyperparmeters.epochs):
        total_loss = 0

        for batch ,(x, y) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(x, y, hidden_state, encoder, decoder, loss_object, optimizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                sample_result, sample, weights = evaluate_training(
                    sample, tokenizer, encoder, decoder, hyperparmeters.hidden_units)
                print(f'Epoch : {epoch + 1}, batch : {batch}, loss : {batch_loss:.5f}, sample : {sample}, result : {sample_result}')





        if hyperparmeters.model_save_path and epoch % hyperparmeters.epochs_per_save == 0:
            time = datetime.now().strftime('%Y%m%d%H%M%S')
            encoder.save_weights(os.path.join(encoder_path, 'encoder_{}.ckpt'.format(time)))
            decoder.save_weights(os.path.join(decoder_path, 'decoder_{}.ckpt'.format(time)))
        sample_result, sample, weights = evaluate_training(
            sample, tokenizer, encoder, decoder, hyperparmeters.hidden_units)
        print(f'Epoch : {epoch + 1}, loss : {total_loss/steps_per_epoch:.5f}, sample : {sample}, result : {sample_result}')











