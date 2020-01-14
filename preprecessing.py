import json, os
import tensorflow as tf
from datetime import datetime
from tensorflow import  keras

class Tokenizer():

    def __init__(self, fit_source):
        assert fit_source == 'file' or 'json'
        self.fit_source = fit_source


    def fit(self, filename, stop_words = [], start_end = True):
        if self.fit_source == 'json':
            print("if fit_source is json , stop_words and start_end can't be supported")
        if self.fit_source == 'file':
            with open(filename, 'r', encoding='utf-8') as f:
                self.vocab = list(sorted(set(list(f.read()))))
                if start_end:
                    self.vocab.append('<bos>')
                    self.vocab.append('<eos>')
                for stop_word in stop_words:
                    self.vocab.remove(stop_word)
                self.atom = dict((char, index) for index, char in enumerate(self.vocab))
        if self.fit_source == 'json':
            with open(filename, 'r', encoding='utf-8') as f:
                self.atom = json.load(f)
                self.vocab = [''] * len(self.atom)
                for k in self.atom:
                    self.vocab[self.atom[k]] = k
        return self.vocab, self.atom

    def idx_to_se(self,indices):
        return ''.join([self.vocab[index] for index in indices])

    def se_to_idx(self, sentence, start_end = False):
        if len(sentence) == 5 or len(sentence) == 7:
            if start_end:
                return [self.atom['<bos>']] + [self.atom[char] for char in sentence] + [self.atom['<eos>']]
            else:
                return [self.atom[char] for char in sentence]
        else:
            sentence = sentence.strip('\n')
            if start_end:
                sentences = sentence.replace('。', '').split('，')
                return [self.atom['<bos>']] + [self.atom[char] for char in sentences[0]] + [self.atom['<eos>']] + [self.atom['<bos>']] + [self.atom[char] for char in sentences[1]] + [self.atom['<eos>']]
            else:
                sentence = sentence.replace('，', '').replace('。', '')
                return [self.atom[char] for char in sentence]

    def save_to_json(self, save_path):
        time = datetime.now().strftime('%Y%m%d%H%M%S')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_path = os.path.join(save_path, time + '.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.atom, f, ensure_ascii=False, indent=4)
        print(f'词典文件已存入{file_path}')



def get_dataset(filename, tokenizer, batch_size, length=7):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if '□' not in line:
                sentences.append(line)
    print(f'共得数据{len(sentences)}条')
    text_as_idx = [tokenizer.se_to_idx(sentence, start_end=True) for sentence in sentences]
    num_examples = len(text_as_idx)
    def split_sentence(sentence):
        return sentence[0: length + 2] , sentence[length + 2 : ]
    dataset = tf.data.Dataset.from_tensor_slices(text_as_idx)
    dataset = dataset.map(split_sentence)
    dataset = dataset.shuffle(10000).batch(batch_size)
    return dataset, num_examples

