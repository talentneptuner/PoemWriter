class HyperParmaters():
    def __init__(self,tokenizer_file, train_filepath, embedding_dims, hidden_units, epochs, batch_size,
                 learning_rate = 0.003, attention_type = 'bahdanau',
                 tokenizer_type='file', stop_words=[], model_save_path = None,
                 epochs_per_save = 10,
                 save_tokenizer = False,
                 tokenizer_save_path = None):

        self.tokenizer_type = tokenizer_type
        self.tokenizer_file = tokenizer_file
        self.train_filepath = train_filepath
        self.stop_words = stop_words
        self.save_tokenizer = save_tokenizer
        self.tokenizer_save_path = tokenizer_save_path

        self.batch_size = batch_size

        self.embedding_dims = embedding_dims
        self.hidden_units = hidden_units
        self.attention_type = attention_type

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.epochs_per_save = epochs_per_save