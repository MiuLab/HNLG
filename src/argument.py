import argparse


def define_arguments(script=False):
    if script:
        nargs = "+"
    else:
        nargs = None

    parser = argparse.ArgumentParser(
            description='Process the data and parameters.')
# PATH
    parser.add_argument(
            '--data_dir', default="../data", nargs=nargs,
            help='the directory of data')
# DATA
    parser.add_argument(
            "--dataset", default="E2ENLG", nargs=nargs,
            choices=["E2ENLG"],
            help='the dataset would be used [E2ENLG]')
    parser.add_argument(
            '--fold_attr', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='fold the attribute into one token or not \
                    (only activate when using E2E dataset) [1]')
    parser.add_argument(
            '--vocab_size', type=int, default=500, nargs=nargs,
            help='the vocab size of tokenizer [500]')
    parser.add_argument(
            '--use_embedding', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='use GloVe embeddings or not [0]')
    parser.add_argument(
            '--regen', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='regenerate the data or not [0]')
    parser.add_argument(
            '--replace_model', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='replace the saved model or not [0]')
    parser.add_argument(
            '--is_spacy', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='use SpaCy as tokenizer or use NLTK instead [1]')
    parser.add_argument(
            '--is_lemma', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='lemmatize or not when parsing the data [1]')
    parser.add_argument(
            '--use_punct', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='use punct or not[0]')
    parser.add_argument(
            '--en_max_length', type=int, default=-1, nargs=nargs,
            help='the max length of encoder input (-1 for no limit) [-1]')
    parser.add_argument(
            '--de_max_length', type=int, default=-1, nargs=nargs,
            help='the max length of decoder output (-1 for no limit) [-1]')
    parser.add_argument(
            '--min_length', type=int, default=5, nargs=nargs,
            help='the min length of label sentence (-1 for no limit) [5]')
    parser.add_argument(
            '--partition_ratio', type=float, default=0.95, nargs=nargs,
            help='the partition ratio of training and testing data [0.95]')
# MODEL
    parser.add_argument(
            '--cell', default='GRU', nargs=nargs,
            choices=["GRU", "LSTM"],
            help='the cell used in RNN [GRU]')
    parser.add_argument(
            '--n_layers', type=int, default=4,
            nargs=nargs, choices=[1, 2, 4],
            help='the hierarchical layers of decoder [2]')
    parser.add_argument(
            '--n_en_layers', type=int, default=1, nargs=nargs,
            help='the number of RNN layers of encoder [1]')
    parser.add_argument(
            '--n_de_layers', type=int, default=1, nargs=nargs,
            help='the number of RNN layers of decoders [1]')
    parser.add_argument(
            '--en_hidden_size', type=int, default=200, nargs=nargs,
            help='the hidden size of encoder RNNs [200]')
    parser.add_argument(
            '--de_hidden_size', type=int, default=100, nargs=nargs,
            help='the hidden size of decoder RNNs [100]')
    parser.add_argument(
            '--en_embedding', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='use embedding in encoder or not [1]')
    parser.add_argument(
            '--en_use_attr_init_state', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='use semantic attributes vector as \
                    encoder initial state or not [0]')
    parser.add_argument(
            '--share_embedding', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='share embedding between encoder and decoder or not \
                    (only activate when using embedding in encoder) [1]')
    parser.add_argument(
            '--embedding_dim', type=int, default=50, nargs=nargs,
            help='the embedding dimension (when only decoder use embeddings \
                    or encoder and decoder use shared embedding) [50]')
    parser.add_argument(
            '--en_embedding_dim', type=int, default=50, nargs=nargs,
            help='the embedding dimension of encoder \
                    (when encoder and decoder use difference embeddings) [50]')
    parser.add_argument(
            '--de_embedding_dim', type=int, default=50, nargs=nargs,
            help='the embedding dimension of decoder \
                    (when encoder and decoder use difference embeddings) [50]')
    parser.add_argument(
            '--attn_method', default='none', nargs=nargs,
            # choices=['concat', 'general', 'dot', 'none'],
            help='the method of attention mechanism [concat]')
    parser.add_argument(
            '--bidirectional', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='bidirectional rnn? [1]')
    parser.add_argument(
            '--feed_last', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='concat last step output of the decoder \
                    to the next step input [1]')
    parser.add_argument(
            '--repeat_input', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='repeat input from the last layer of the decoder \
                    when output does not match [1]'
    )
    parser.add_argument(
            '--batch_norm', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='add batch normalization layer between \
                    hidden layer and vocab output [0]')
# TRAINING
    parser.add_argument(
            '--epochs', type=int, default=20, nargs=nargs,
            help='train for N epochs [20]')
    parser.add_argument(
            '--batch_size', type=int, default=32, nargs=nargs,
            help='the size of batch [32]')
    parser.add_argument(
            '--en_optimizer', default="Adam", nargs=nargs,
            choices=['Adam', 'RMSprop', 'SGD'],
            help='the optimizer of encoder (Adam / RMSprop / SGD) [Adam]')
    parser.add_argument(
            '--de_optimizer', default="Adam", nargs=nargs,
            choices=['Adam', 'RMSprop', 'SGD'],
            help='the optimizer of decoder (Adam / RMSProp / SGD) [Adam]')
    parser.add_argument(
            '--en_learning_rate', type=float, default=1e-3, nargs=nargs,
            help='the learning rate of encoder [1e-3]')
    parser.add_argument(
            '--de_learning_rate', type=float, default=1e-3, nargs=nargs,
            help='the learning rate of decoders [1e-3]')
    parser.add_argument(
            '--split_teacher_forcing', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='split inner and inter teacher forcing [0]')
    parser.add_argument(
            '--teacher_forcing_ratio', type=float, default=0.5, nargs=nargs,
            help='the ratio of teacher forcing [0.5]')
    parser.add_argument(
            '--inner_teacher_forcing_ratio', type=float, default=0.5,
            nargs=nargs,
            help='the ratio of inter teacher forcing [0.5]')
    parser.add_argument(
            '--inter_teacher_forcing_ratio', type=float, default=0.5,
            nargs=nargs,
            help='the ratio of inter teacher forcing [0.5]')
    parser.add_argument(
            '--tf_decay_rate', type=float, default=0.9, nargs=nargs,
            help='the ratio of teacher forcing decay rate [0.9]')
    parser.add_argument(
            '--inner_tf_decay_rate', type=float, default=0.9,
            nargs=nargs,
            help='the ratio of inter teacher forcing decay rate [0.9]')
    parser.add_argument(
            '--inter_tf_decay_rate', type=float, default=0.9,
            nargs=nargs,
            help='the ratio of inter teacher forcing decay rate [0.9]')
    parser.add_argument(
            '--schedule_sampling', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help="use schedule sampling or not [0]")
    parser.add_argument(
            '--inner_schedule_sampling', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help="use inner schedule sampling or not [1]")
    parser.add_argument(
            '--inter_schedule_sampling', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help="use inter schedule sampling or not [1]")
    parser.add_argument(
            '--is_curriculum', type=int, default=1,
            nargs=nargs, choices=range(0, 2),
            help='use curriculum learning or not [1]')
    parser.add_argument(
            '--padding_loss', type=float, default=0.0, nargs=nargs,
            help='the weight of padding loss [0.0]')
    parser.add_argument(
            '--eos_loss', type=float, default=1.0, nargs=nargs,
            help='the weight of EOS loss [2.0]')
    parser.add_argument(
            '--max_norm', type=float, default=0.25, nargs=nargs,
            help='max norm during training [0.25]')
    parser.add_argument(
            '--finetune_embedding', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='finetune pre-trained word embedding \
                    during training or not [0]')
# VERBOSE, VALIDATION AND SAVE
    parser.add_argument(
            '--verbose_level', type=int, default=1,
            nargs=nargs, choices=range(0, 3),
            help='the verbose level of config (from 0 to 2) [1]')
    parser.add_argument(
            '--verbose_epochs', type=int, default=0, nargs=nargs,
            help='verbose every N epochs (0 for every iters) [0]')
    parser.add_argument(
            '--verbose_batches', type=int, default=500, nargs=nargs,
            help='verbose every N batches [100]')
    parser.add_argument(
            '--valid_epochs', type=int, default=1, nargs=nargs,
            help='run validation batch every N epochs [1]')
    parser.add_argument(
            '--valid_batches', type=int, default=20, nargs=nargs,
            help='run validation batch every N batches [20]')
    parser.add_argument(
            '--save_epochs', type=int, default=1, nargs=nargs,
            help='save model every N epochs [1]')
    parser.add_argument(
            '--is_load', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='load saved model or not [0]')
    parser.add_argument(
            '--check_mem_usage_batches', type=int, default=0, nargs=nargs,
            help='check GPU memory usage every N batches (0 for never) [0]')
    parser.add_argument(
            '--dir_name', type=str, default='test',
            help='log dir name')
    parser.add_argument(
            '--h_attn', type=int, default=0,
            nargs=nargs, choices=range(0, 2),
            help='load saved model or not [0]')
    args = parser.parse_args()
    return(parser, args)
