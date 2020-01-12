from preprecessing import Tokenizer, get_dataset
from module import Seq2SeqEncoder, Seq2SeqAttention, Seq2SeqDecoder
import tensorflow as tf

if __name__ == '__main__':
    tokenizer = Tokenizer('file')
    tokenizer.fit('./data/tang_poems_7.txt', stop_words=['□', '\n', '，', '。'])
    dataset, _ = get_dataset('./data/tang_poems_7.txt', tokenizer, batch_size=2)
    # encoder = Seq2SeqEncoder(len(tokenizer.vocab), 256, 1024)
    # hidden_state = encoder.initial_hidden_state(2)
    # decoder = Seq2SeqDecoder(len(tokenizer.vocab), 256, 1024)
    for x, y in dataset.take(1):
        print(x, y)
        print(tokenizer.idx_to_se(x[0]), tokenizer.idx_to_se(y[0]))
        # output, state = encoder(x, hidden_state)
        # outputs = decoder(tf.random.uniform((2, 1)), state, output)
        # for martix in outputs:
        #     print(martix.shape)

