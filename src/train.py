
import argparse
import pickle
from model import NLG
from data_engine import DataEngine
from text_token import _UNK, _PAD, _BOS, _EOS
import torch
import torch.nn as nn
import numpy as np
import os
from utils import print_config, add_path
from model_utils import get_embeddings
from argument import define_arguments
from utils import get_time

_, args = define_arguments()

args = add_path(args)
if args.verbose_level > 0:
    print_config(args)

use_cuda = torch.cuda.is_available()
train_data_engine = DataEngine(
    data_dir=args.data_dir,
    dataset=args.dataset,
    save_path=args.train_data_file,
    vocab_path=args.vocab_file,
    is_spacy=args.is_spacy,
    is_lemma=args.is_lemma,
    fold_attr=args.fold_attr,
    use_punct=args.use_punct,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    en_max_length=(args.en_max_length if args.en_max_length != -1 else None),
    de_max_length=(args.de_max_length if args.de_max_length != -1 else None),
    regen=args.regen,
    train=True
)

test_data_engine = DataEngine(
    data_dir=args.data_dir,
    dataset=args.dataset,
    save_path=args.valid_data_file,
    vocab_path=args.vocab_file,
    is_spacy=args.is_spacy,
    is_lemma=args.is_lemma,
    fold_attr=args.fold_attr,
    use_punct=args.use_punct,
    vocab_size=args.vocab_size,
    n_layers=args.n_layers,
    en_max_length=(args.en_max_length if args.en_max_length != -1 else None),
    de_max_length=(args.de_max_length if args.de_max_length != -1 else None),
    regen=args.regen,
    train=False
)

if train_data_engine.split_vocab:
    vocab, rev_vocab, token_vocab, rev_token_vocab = \
            pickle.load(open(args.vocab_file, 'rb'))
    en_vocab_size = len(token_vocab)
    de_vocab_size = vocab_size = args.vocab_size + 4
else:
    en_vocab_size = de_vocab_size = vocab_size = args.vocab_size + 4

if args.dataset == "REPEATSEQ":
    args.n_layers = 1
    args.use_embedding = 0

if args.n_layers == 1:
    args.is_curriculum = 0

if args.use_embedding:
    embedding_dim = (
            args.embedding_dim if (args.en_embedding and args.share_embedding)
            else args.de_embedding_dim)
    embeddings = get_embeddings(vocab, args.embeddings_dir, embedding_dim)
else:
    embeddings = None


model = NLG(
        n_decoders=args.n_layers,
        cell=args.cell,
        n_en_layers=args.n_en_layers,
        n_de_layers=args.n_de_layers,
        bidirectional=args.bidirectional,
        feed_last=args.feed_last,
        repeat_input=args.repeat_input,
        batch_norm=args.batch_norm,
        vocab_size=vocab_size,
        en_vocab_size=en_vocab_size,
        de_vocab_size=de_vocab_size,
        embedding_dim=args.embedding_dim,
        en_embedding_dim=args.en_embedding_dim,
        de_embedding_dim=args.de_embedding_dim,
        en_hidden_size=args.en_hidden_size,
        de_hidden_size=args.de_hidden_size,
        batch_size=args.batch_size,
        en_optimizer=args.en_optimizer,
        de_optimizer=args.de_optimizer,
        en_learning_rate=args.en_learning_rate,
        de_learning_rate=args.de_learning_rate,
        attn_method=args.attn_method,
        train_data_engine=train_data_engine,
        test_data_engine=test_data_engine,
        use_embedding=args.use_embedding,
        en_use_attr_init_state=args.en_use_attr_init_state,
        embeddings=embeddings,
        en_embedding=args.en_embedding,
        share_embedding=args.share_embedding,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        is_load=args.is_load,
        check_mem_usage_batches=args.check_mem_usage_batches,
        replace_model=args.replace_model,
        finetune_embedding=args.finetune_embedding,
        model_config=args
)


loss_weight = np.ones(args.vocab_size + 4)
loss_weight[_PAD] = args.padding_loss
loss_weight[_EOS] = args.eos_loss
loss_weight = torch.FloatTensor(loss_weight)
loss_weight = loss_weight.cuda() if use_cuda else loss_weight
loss_func = nn.CrossEntropyLoss(weight=loss_weight)

if args.is_curriculum:
    for N in range(1, args.n_layers+1):
        model.train(
                epochs=args.epochs // args.n_layers,
                batch_size=args.batch_size,
                criterion=loss_func,
                verbose_epochs=args.verbose_epochs,
                verbose_batches=args.verbose_batches,
                valid_epochs=args.valid_epochs,
                valid_batches=args.valid_batches,
                save_epochs=args.save_epochs,
                split_teacher_forcing=args.split_teacher_forcing,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                inner_teacher_forcing_ratio=args.inner_teacher_forcing_ratio,
                inter_teacher_forcing_ratio=args.inter_teacher_forcing_ratio,
                tf_decay_rate=args.tf_decay_rate,
                inner_tf_decay_rate=args.inner_tf_decay_rate,
                inter_tf_decay_rate=args.inter_tf_decay_rate,
                schedule_sampling=args.schedule_sampling,
                inner_schedule_sampling=args.inner_schedule_sampling,
                inter_schedule_sampling=args.inter_schedule_sampling,
                max_norm=args.max_norm,
                curriculum_layers=N)

else:
    model.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            criterion=loss_func,
            verbose_epochs=args.verbose_epochs,
            verbose_batches=args.verbose_batches,
            valid_epochs=args.valid_epochs,
            valid_batches=args.valid_batches,
            save_epochs=args.save_epochs,
            split_teacher_forcing=args.split_teacher_forcing,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            inner_teacher_forcing_ratio=args.inner_teacher_forcing_ratio,
            inter_teacher_forcing_ratio=args.inter_teacher_forcing_ratio,
            tf_decay_rate=args.tf_decay_rate,
            inner_tf_decay_rate=args.inner_tf_decay_rate,
            inter_tf_decay_rate=args.inter_tf_decay_rate,
            schedule_sampling=args.schedule_sampling,
            inner_schedule_sampling=args.inner_schedule_sampling,
            inter_schedule_sampling=args.inter_schedule_sampling,
            max_norm=args.max_norm,
            curriculum_layers=args.n_layers)
