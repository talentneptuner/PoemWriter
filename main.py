from train import *
from hyperParameters import HyperParmaters

if __name__ == '__main__':
    hyperparameters = HyperParmaters(tokenizer_file='./data/tang_poems_7.txt',
                                     train_filepath='./data/tang_poems_7.txt',
                                     embedding_dims= 256,
                                     hidden_units=1024,
                                     epochs = 10,
                                     batch_size=64,
                                     save_tokenizer=True,
                                     epochs_per_save=1,
                                     tokenizer_save_path='./tokeinzer',
                                     model_save_path='./model')
    train(hyperparameters)