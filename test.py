from train import *
from hyperParameters import HyperParmaters

if __name__ == '__main__':
    hyperparameters = HyperParmaters(tokenizer_type='json',
                                     tokenizer_file='./tokeinzer/20200113125627.json',
                                     train_filepath=None,
                                     embedding_dims= 256,
                                     hidden_units=1024,
                                     epochs = 10,
                                     batch_size=64,
                                     model_save_path='./model',
                                     sample_display='万里悲秋常作客',
                                     attention_type='google')
    evaluate_sample(hyperparameters)