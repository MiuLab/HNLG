import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm

import random
import numpy as np
import os

from module import EncoderRNN
from module import DecoderRNN
# from utils import single_BLEU, BLEU, single_ROUGE, ROUGE, best_ROUGE, print_time_info, check_dir, print_curriculum_status
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn, build_optimizer
from logger import Logger

from tqdm import tqdm

use_cuda = torch.cuda.is_available()


class NLG:
    def __init__(
            self,
            batch_size,
            en_optimizer,
            de_optimizer,
            en_learning_rate,
            de_learning_rate,
            attn_method,
            train_data_engine,
            test_data_engine,
            use_embedding,
            en_use_attr_init_state,
            en_hidden_size=100,
            de_hidden_size=100,
            en_vocab_size=None,
            de_vocab_size=None,
            vocab_size=None,
            en_embedding_dim=None,
            de_embedding_dim=None,
            embedding_dim=None,
            embeddings=None,
            en_embedding=True,
            share_embedding=True,
            n_decoders=2,
            cell="GRU",
            n_en_layers=1,
            n_de_layers=1,
            bidirectional=False,
            feed_last=False,
            repeat_input=False,
            batch_norm=False,
            model_dir="./model",
            log_dir="./log",
            is_load=True,
            check_mem_usage_batches=0,
            replace_model=True,
            finetune_embedding=False,
            model_config=None
    ):

        # Initialize attributes
        self.data_engine = train_data_engine
        self.check_mem_usage_batches = check_mem_usage_batches
        self.n_decoders = n_decoders
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.en_embedding_dim = en_embedding_dim
        self.de_embedding_dim = de_embedding_dim
        self.embedding_dim = embedding_dim
        self.repeat_input = repeat_input
        self.de_hidden_size = de_hidden_size
        self.bidirectional = bidirectional
        self.dir_name = model_config.dir_name
        self.h_attn = model_config.h_attn


        # embedding layer setting
        if not en_embedding:
            en_embed = None
            de_embed = nn.Embedding(de_vocab_size, de_embedding_dim)
            if use_embedding:
                de_embed.weight = embeddings
                if not finetune_embedding:
                    de_embed.weight.requires_grad = False
        else:
            if share_embedding:
                embed = nn.Embedding(vocab_size, embedding_dim)
                if use_embedding:
                    embed.weight = embeddings
                    if not finetune_embedding:
                        embed.weight.requires_grad = False
                en_embed = embed
                de_embed = embed
            else:
                en_embed = nn.Embedding(en_vocab_size, en_embedding_dim)
                de_embed = nn.Embedding(de_vocab_size, de_embedding_dim)
                if use_embedding:
                    # in E2ENLG dataset, only decoder use word embedding
                    de_embed.weight = embeddings
                    if not finetune_embedding:
                        de_embed.weight.requires_grad = False

        self.encoder = EncoderRNN(
                en_embedding=en_embedding,
                embedding=en_embed,
                en_vocab_size=en_vocab_size,
                en_embedding_dim=(
                    embedding_dim
                    if share_embedding and en_embedding
                    else en_embedding_dim),
                hidden_size=en_hidden_size,
                n_layers=n_en_layers,
                bidirectional=bidirectional,
                cell=cell)

        self.cell = cell
        self.decoders = []
        for n in range(n_decoders):
            decoder = DecoderRNN(
                    embedding=de_embed,
                    de_vocab_size=de_vocab_size,
                    de_embedding_dim=(
                        embedding_dim
                        if share_embedding and en_embedding
                        else self.de_embedding_dim),
                    en_hidden_size=en_hidden_size,
                    de_hidden_size=de_hidden_size,
                    n_en_layers=n_en_layers,
                    n_de_layers=n_de_layers,
                    bidirectional=bidirectional,
                    feed_last=(True
                               if feed_last and n > 0
                               else False),
                    batch_norm=batch_norm,
                    attn_method=attn_method,
                    cell=cell,
                    h_attn=self.h_attn,
                    index=n
            )
            self.decoders.append(decoder)

        self.encoder = self.encoder.cuda() if use_cuda else self.encoder
        self.decoders = [
                decoder.cuda()
                if use_cuda else decoder for decoder in self.decoders]

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine
        self.train_data_loader = DataLoader(
                train_data_engine,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True)

        self.test_data_loader = DataLoader(
                test_data_engine,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                collate_fn=collate_fn,
                pin_memory=True)

        # encoder parameters optimization
        self.encoder_parameters = filter(
                lambda p: p.requires_grad, self.encoder.parameters())
        self.encoder_optimizer = build_optimizer(
                en_optimizer, self.encoder_parameters,
                en_learning_rate)
        # decoder parameters optimization
        decoder_parameters = []
        for decoder in self.decoders:
            decoder_parameters.extend(list(decoder.parameters()))
        self.decoder_parameters = filter(
                lambda p: p.requires_grad, decoder_parameters)
        self.decoder_optimizer = build_optimizer(
                de_optimizer, self.decoder_parameters,
                de_learning_rate)

        print_time_info("Model create complete")

        if not replace_model:
            self.model_dir = os.path.join(
                self.model_dir,
                self.dir_name
            )

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            if not is_load:
                check_dir(self.model_dir)
        self.log_dir = os.path.join(
            self.log_dir,
            self.dir_name
        )


        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(os.path.join(self.log_dir, "validation"))

        if not is_load:
            with open(os.path.join(self.log_dir, "model_config"), "w+") as f:
                for arg in vars(model_config):
                    f.write("{}: {}\n".format(
                        arg, str(getattr(model_config, arg))))
                f.close()

        if is_load:
            self.load_model(self.model_dir)

        # Initialize the log files
        self.logger = Logger(self.log_dir)
        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_batch_log_path = os.path.join(
                self.log_dir, "valid_batch_log.csv")
        self.valid_epoch_log_path = os.path.join(
                self.log_dir, "valid_epoch_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch, batch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")
        with open(self.valid_batch_log_path, 'w') as file:
            file.write("epoch, batch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")
        with open(self.valid_epoch_log_path, 'w') as file:
            file.write("epoch, loss, avg-bleu, avg-rouge(1,2,L,BE)\n")

        # Initialize batch count
        self.batches = 0
        self.en_use_attr_init_state = en_use_attr_init_state

    def train(self, epochs, batch_size, criterion, verbose_epochs=1,
              verbose_batches=1, valid_epochs=1, valid_batches=1000,
              save_epochs=10,
              split_teacher_forcing=False,
              teacher_forcing_ratio=0.5,
              inner_teacher_forcing_ratio=0.5,
              inter_teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              inner_tf_decay_rate=0.9,
              inter_tf_decay_rate=0.9,
              schedule_sampling=False,
              inner_schedule_sampling=False,
              inter_schedule_sampling=False,
              max_norm=0.25, curriculum_layers=2):

        self.batches = 0
        print_curriculum_status(curriculum_layers)

        _teacher_forcing_ratio = teacher_forcing_ratio
        _inner_teacher_forcing_ratio = inner_teacher_forcing_ratio
        _inter_teacher_forcing_ratio = inter_teacher_forcing_ratio

        for idx in range(1, epochs+1):
            epoch_loss = 0
            epoch_BLEU = 0
            epoch_ROUGE = np.array([0, 0, 0, 0])

            # training
            print("+----------------+")
            print("|    TRAINING    |")
            print("+----------------+")
            for b_idx, batch in enumerate(tqdm(self.train_data_loader)):
                self.batches += 1
                # test_loss, test_single_BLEU, test_BLEU, test_single_ROUGE, test_ROUGE, test_best_ROUGE
                # batch_loss, batch_BLEU, batch_ROUGE = self.run_batch(
                batch_loss, batch_single_BLEU, batch_BLEU, batch_single_ROUGE, batch_ROUGE, batch_best_ROUGE = self.run_batch(
                        batch,
                        criterion,
                        curriculum_layers,
                        testing=False,
                        split_teacher_forcing=split_teacher_forcing,
                        teacher_forcing_ratio=_teacher_forcing_ratio,
                        inner_teacher_forcing_ratio=(
                            _inner_teacher_forcing_ratio),
                        inter_teacher_forcing_ratio=(
                            _inter_teacher_forcing_ratio),
                        schedule_sampling=schedule_sampling,
                        inner_schedule_sampling=inner_schedule_sampling,
                        inter_schedule_sampling=inter_schedule_sampling,
                        max_norm=max_norm
                )

            # save model
            if idx % save_epochs == 0:
                print_time_info("Epoch {}: save model...".format(idx))
                self.save_model(self.model_dir)

            _teacher_forcing_ratio *= tf_decay_rate
            _inner_teacher_forcing_ratio *= inner_tf_decay_rate
            _inter_teacher_forcing_ratio *= inter_tf_decay_rate

    def test(
            self,
            epochs,
            batch_size,
            criterion,
            verbose_epochs=1,
            verbose_batches=1,
            valid_epochs=1,
            valid_batches=1000,
            save_epochs=10,
            split_teacher_forcing=False,
            teacher_forcing_ratio=0.5,
            inner_teacher_forcing_ratio=0.5,
            inter_teacher_forcing_ratio=0.5,
            tf_decay_rate=0.9,
            inner_tf_decay_rate=0.9,
            inter_tf_decay_rate=0.9,
            schedule_sampling=False,
            inner_schedule_sampling=False,
            inter_schedule_sampling=False,
            max_norm=0.25,
            curriculum_layers=2):

        self.batches = 0

        print("+----------------+")
        print("|     TESTING    |")
        print("+----------------+")
        # batch = next(iter(self.test_data_loader))
        avg_test_loss = 0
        avg_test_single_BLEU = 0
        avg_test_BLEU = 0
        avg_test_single_ROUGE = 0
        avg_test_ROUGE = 0
        avg_test_best_ROUGE = 0
        batch_amount = 0

        for batch_idx, batch in enumerate(tqdm(self.test_data_loader)):
            test_loss, test_single_BLEU, test_BLEU, test_single_ROUGE, test_ROUGE, test_best_ROUGE = self.run_batch(
                    batch,
                    criterion,
                    curriculum_layers,
                    testing=True,
                    result_path=os.path.join(
                        os.path.join(self.log_dir, "validation"),
                        "test.txt"
                    )
            )
            avg_test_loss += test_loss
            avg_test_single_BLEU += test_single_BLEU
            avg_test_BLEU += test_BLEU
            avg_test_single_ROUGE += test_single_ROUGE
            avg_test_ROUGE += test_ROUGE
            avg_test_best_ROUGE += test_best_ROUGE
            batch_amount = batch_idx + 1

        avg_test_loss = (avg_test_loss/batch_amount)
        avg_test_single_BLEU = (avg_test_single_BLEU/batch_amount)
        avg_test_BLEU = (avg_test_BLEU/batch_amount)
        avg_test_single_ROUGE = (avg_test_single_ROUGE/batch_amount)
        avg_test_ROUGE = (avg_test_ROUGE/batch_amount)
        avg_test_best_ROUGE = (avg_test_best_ROUGE/batch_amount)

        print(avg_test_single_BLEU)
        print(avg_test_BLEU)
        print(avg_test_single_ROUGE)
        print(avg_test_ROUGE)
        print(avg_test_best_ROUGE)

        with open("test_results.txt", 'a') as file:
            file.write("{}\n".format(self.dir_name))
            file.write("{}\n{}\n{}\n{}\n{}\n".format(
                avg_test_single_BLEU,
                avg_test_BLEU,
                ', '.join(map(str, avg_test_single_ROUGE)),
                ', '.join(map(str, avg_test_ROUGE)),
                ', '.join(map(str, avg_test_best_ROUGE))
            ))
            file.write("{}\t{}\t{}\t{}\t{}\t\n".format(
                avg_test_single_BLEU,
                avg_test_BLEU,
                '\t'.join(map(str, avg_test_single_ROUGE)),
                '\t'.join(map(str, avg_test_ROUGE)),
                '\t'.join(map(str, avg_test_best_ROUGE))
            ))

    def run_batch(
            self,
            batch,
            criterion,
            curriculum_layers,
            testing=False,
            split_teacher_forcing=False,
            teacher_forcing_ratio=0.5,
            inner_teacher_forcing_ratio=0.5,
            inter_teacher_forcing_ratio=0.5,
            schedule_sampling=False,
            inner_schedule_sampling=False,
            inter_schedule_sampling=False,
            max_norm=None,
            result_path=None
            ):
        """
        When testing=False, run_batch is in training mode, and you should
        pass the argument teacher_forcing_ratio and max_norm
        When testing=True, run_batch is in testing mode, you should pass
        the argument result_path to store the result of testing batch
        """
        if not testing:
            # Initialize the optimizers (when training)
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

        # Initialize the loss
        loss = 0
        all_loss = 0

        # Generate the inputs for encoder
        if len(batch) == 3:
            raw_encoder_input, decoder_labels, de_lengths = batch
        else:
            raw_encoder_input, decoder_labels, de_lengths, refs, sf_data = batch

        batch_size = len(raw_encoder_input)
        encoder_input = Variable(
                torch.LongTensor(raw_encoder_input),
                volatile=testing)
        encoder_input = encoder_input.cuda() if use_cuda else encoder_input
        encoder_hidden = (self.encoder.initAttrHidden(
            encoder_input, batch_size)
            if self.en_use_attr_init_state
            else self.encoder.initHidden(batch_size))

        # We need to initialize the cell states for LSTM
        if self.cell == "LSTM":
            encoder_cell = self.encoder.initHidden(batch_size)

        if self.encoder.cell == "GRU":
            encoder_outputs, encoder_hidden = self.encoder(
                    encoder_input, encoder_hidden)
        elif self.encoder.cell == "LSTM":
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
                    encoder_input, encoder_hidden, encoder_cell)


        # For first layer of decoder, we use the last output as input;
        # for the other layers, we use the output from previous layer as input
        # Prepare the space for results from decoder (when testing)
        if testing:
            decoder_results = [np.zeros(
                (batch_size, decoder_labels[idx].shape[1]))
                for idx in range(curriculum_layers)]

        if not testing:
            if not split_teacher_forcing:
                if schedule_sampling:
                    use_teacher_forcing = [
                                [
                                    True if random.random() <
                                    teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_teacher_forcing = (
                            True if random.random() < teacher_forcing_ratio
                            else False)
                    use_teacher_forcing = [
                                [
                                    _use_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                use_inner_teacher_forcing = use_teacher_forcing
                use_inter_teacher_forcing = [[0]] + use_teacher_forcing[:-1]
            else:
                if inner_schedule_sampling:
                    use_inner_teacher_forcing = [
                                [
                                    True if random.random() <
                                    inner_teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_inner_teacher_forcing = (
                            True
                            if random.random() < inner_teacher_forcing_ratio
                            else False)
                    use_inner_teacher_forcing = [
                                [
                                    _use_inner_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                if inter_schedule_sampling:
                    use_inter_teacher_forcing = [
                                [
                                    True if random.random() <
                                    inter_teacher_forcing_ratio
                                    else False
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers)
                            ]
                else:
                    _use_inter_teacher_forcing = (
                            True
                            if random.random() < inter_teacher_forcing_ratio
                            else False)
                    use_inter_teacher_forcing = [
                                [
                                    _use_inter_teacher_forcing
                                    for _ in range(
                                        decoder_labels[idx].shape[1])
                                    ]
                                for idx in range(curriculum_layers - 1)
                            ]
                    use_inter_teacher_forcing = \
                        [[0]] + use_inter_teacher_forcing
        else:
            use_teacher_forcing = [
                        [
                            False
                            for _ in range(decoder_labels[idx].shape[1])
                            ]
                        for idx in range(curriculum_layers)
                    ]
            use_inner_teacher_forcing = use_teacher_forcing
            use_inter_teacher_forcing = [[0]] + use_teacher_forcing[:-1]

        # BLEU/ROUGE scores
        bleu_scores = []
        rouge_scores = []

        """
            First layer: seq2seq
            Other layers: RNN (input from first layer output / labels)
        """
        # all_decoder_inputs: input from the last layer
        # real_decoder_inputs: actual input from the last layer
        # note that there is 'repeat input' mechanism
        all_decoder_inputs = [[] for _ in range(curriculum_layers)]
        real_decoder_inputs = [[] for _ in range(curriculum_layers)]
        decoder_inputs = None
        last_decoder_hiddens = None
        for d_idx, decoder in enumerate(self.decoders[:curriculum_layers]):
            # for recording actual inputs
            _real_decoder_inputs = Variable(
                    torch.LongTensor(batch_size, 1).fill_(_PAD))
            _real_decoder_inputs = (
                _real_decoder_inputs.cuda()
                if use_cuda else _real_decoder_inputs)
            # Prepare for initial hidden state and cell
            decoder_hidden = decoder.transform_layer(encoder_hidden)
            if self.cell == "LSTM":
                decoder_cell = encoder_cell

            # Prepare for first input of certain layer
            if d_idx == 0:
                # First input of first layer must be _BOS
                decoder_input = Variable(
                        torch.LongTensor(batch_size, 1).fill_(_BOS))
            else:
                if use_inter_teacher_forcing[d_idx][0]:
                    decoder_input = Variable(
                        torch.LongTensor(decoder_labels[d_idx-1][:, 0])) \
                            .unsqueeze(1)
                else:
                    decoder_input = decoder_inputs[:, 0].unsqueeze(1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            _real_decoder_inputs = torch.cat((
                _real_decoder_inputs, decoder_input), 1)
            # Set last_output of the first step to BOS
            last_output = Variable(
                   torch.LongTensor(batch_size, 1).fill_(_BOS)
            )

            # Prepare for input of next layer
            if d_idx < curriculum_layers - 1:
                next_decoder_inputs = Variable(
                        torch.LongTensor(
                            batch_size,
                            decoder_labels[d_idx+1].shape[1]).fill_(_PAD))
                next_decoder_inputs = (
                        next_decoder_inputs.cuda() if use_cuda
                        else next_decoder_inputs)
                # for h attention
                last_decoder_hiddens_temp = Variable(
                        torch.FloatTensor(
                            batch_size,
                            decoder_labels[d_idx+1].shape[1],
                            self.de_hidden_size*(self.bidirectional+1)
                        ).fill_(0))

                last_decoder_hiddens_temp = (
                        last_decoder_hiddens_temp.cuda() if use_cuda
                        else last_decoder_hiddens_temp)

            # for calculating BLEU
            hypothesis = torch.LongTensor(batch_size, 1, 1).fill_(_PAD)
            hypothesis = hypothesis.cuda() if use_cuda else hypothesis

            # Decoding sequence
            # for repeat_input, remember the offset of the label.
            label_idx = [0 for _ in range(batch_size)]
            for idx in range(decoder_labels[d_idx].shape[1]):
                last_output = last_output.cuda() if use_cuda else last_output

                if decoder.cell == "GRU":
                    decoder_output, decoder_hidden = decoder(
                        decoder_input,
                        decoder_hidden,
                        encoder_outputs,
                        last_output=last_output,
                        last_decoder_hiddens=last_decoder_hiddens
                    )
                elif decoder.cell == "LSTM":
                    decoder_output, decoder_hidden, decoder_cell = decoder(
                            decoder_input, decoder_hidden,
                            encoder_outputs, decoder_cell,
                            last_output=last_output)

                target = Variable(
                        torch.from_numpy(
                            decoder_labels[d_idx][:, idx])).cuda()

                loss += criterion(decoder_output.squeeze(1), target)

                topv, topi = decoder_output.data.topk(1)
                hypothesis = torch.cat((hypothesis, topi), 1)
                if d_idx < curriculum_layers - 1:
                    # last_decoder_hiddens[:, idx] = decoder_hidden
                    last_decoder_hiddens_temp[:, idx, :] = decoder_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
                    if use_inter_teacher_forcing[d_idx+1][idx]:
                        next_decoder_inputs[:, idx] = \
                            Variable(torch.from_numpy(
                                decoder_labels[d_idx][:, idx])).cuda()
                    else:
                        next_decoder_inputs[:, idx] = topi

                if testing:
                    decoder_results[d_idx][:, idx] = \
                        topi.cpu().numpy().squeeze((1, 2))
                # Decide next input of decoder
                if idx != decoder_labels[d_idx].shape[1] - 1:
                    # input from last step
                    if use_inner_teacher_forcing[d_idx][idx+1]:
                        last_output = target.unsqueeze(1)
                    else:
                        last_output = Variable(topi).squeeze(1)
                    # input from last layer
                    if d_idx == 0:
                        decoder_input = Variable(topi).squeeze(1)
                    else:
                        if self.repeat_input:
                            decoder_input = np.zeros(
                                    (batch_size, 1), dtype=np.int64)
                            predicts = topi.cpu().numpy().squeeze((1, 2))
                            labels = decoder_inputs.data.cpu().numpy()
                            for b_idx in range(len(label_idx)):
                                #  print(labels[b_idx][label_idx[b_idx]])
                                if predicts[b_idx] == labels[b_idx][
                                        label_idx[b_idx]]:
                                    label_idx[b_idx] += 1
                                decoder_input[b_idx][0] = labels[b_idx][
                                        label_idx[b_idx]]
                            decoder_input = Variable(
                                        torch.LongTensor(decoder_input))
                        else:
                            if idx >= decoder_labels[d_idx-1].shape[1]:
                                decoder_input = Variable(
                                    torch.LongTensor(
                                        batch_size, 1).fill_(_PAD))
                            else:
                                decoder_input = \
                                    decoder_inputs[:, idx+1].unsqueeze(1)

                    decoder_input = (
                            decoder_input.cuda() if use_cuda
                            else decoder_input)

                    _real_decoder_inputs = torch.cat((
                        _real_decoder_inputs, decoder_input), 1)

            if d_idx < curriculum_layers - 1:
                decoder_inputs = next_decoder_inputs
                last_decoder_hiddens = last_decoder_hiddens_temp

                # record the layer inputs
                all_decoder_inputs[d_idx+1] = decoder_inputs.data.cpu().numpy()
                real_decoder_inputs[d_idx+1] = _real_decoder_inputs.data \
                    .cpu().numpy()[:, 1:]

            hypothesis = hypothesis.squeeze(2).cpu().numpy()[:, 1:]

            avg_single_bleu = 0
            avg_bleu = 0
            avg_single_rouge_1_2_l_be = 0
            avg_rouge_1_2_l_be = 0
            avg_best_rouge_1_2_l_be = 0

            if testing:
                single_bleu_score = single_BLEU(decoder_labels[d_idx], hypothesis)
                bleu_score = BLEU(refs, hypothesis)
                avg_single_bleu = np.mean(single_bleu_score)
                avg_bleu = np.mean(bleu_score)
                bleu_scores.append(bleu_score)

                single_rouge_score = single_ROUGE(decoder_labels[d_idx], hypothesis)
                avg_single_rouge_1_2_l_be = np.mean(single_rouge_score, axis=0)
                rouge_score = ROUGE(refs, hypothesis)
                avg_rouge_1_2_l_be = np.mean(rouge_score, axis=0)
                rouge_scores.append(rouge_score)
                best_rouge_score = best_ROUGE(refs, hypothesis)
                avg_best_rouge_1_2_l_be = np.mean(best_rouge_score, axis=0)

            # to prevent the graph keeping growing bigger and resulting in OOM
            # compute the gradients every layer (when training)
            if not testing:
                loss.backward(retain_graph=True)
            all_loss += loss.data[0] / de_lengths[d_idx]
            loss = 0

        if not testing:
            clip_grad_norm(self.encoder_parameters, max_norm)
            self.encoder_optimizer.step()
            clip_grad_norm(self.decoder_parameters, max_norm)
            self.decoder_optimizer.step()

        else:
            # untokenize the sentence,
            # e.g. [100, 200, 300] -> ['trouble', 'fall', 'piece']
            encoder_input = [
                    self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
                    for idx, sent in enumerate(raw_encoder_input)]
            decoder_results = [
                    [self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
                        for sent in decoder_result]
                    for idx, decoder_result in enumerate(decoder_results)]
            decoder_labels = [
                    [self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
                        for sent in decoder_label]
                    for idx, decoder_label in enumerate(decoder_labels)]
            # decoder inputs
            real_decoder_inputs = [
                [self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
                 for sent in real_decoder_input]
                for idx, real_decoder_input in enumerate(real_decoder_inputs)
            ]

            all_decoder_inputs = [
                [self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
                 for sent in all_decoder_input]
                for idx, all_decoder_input in enumerate(all_decoder_inputs)
            ]

            # write test results into files
            with open(result_path, 'a') as file:
                for idx in range(batch_size):
                    file.write("---------\n")
                    file.write("Data {}\n".format(idx))
                    file.write("encoder input: {}\n\n".format(
                        ' '.join(encoder_input[idx])))
                    for d_idx in range(curriculum_layers):
                        file.write("decoder layer {}\n".format(d_idx))
                        if d_idx > 0:
                            file.write(
                                "input from the last layer:\n{}\n".format(
                                    ' '.join(all_decoder_inputs[d_idx][idx])))
                            file.write("actual input:\n{}\n".format(
                                ' '.join(real_decoder_inputs[d_idx][idx])))
                        file.write("prediction:\n{}\n".format(
                            ' '.join(decoder_results[d_idx][idx])))
                        file.write("labels:\n{}\n".format(
                            ' '.join(decoder_labels[d_idx][idx])))
                        file.write("BLEU score: {}\n".format(
                            str(bleu_scores[d_idx][idx])))
                        file.write("ROUGE_(1,2,L,BE): {}\n".format(
                            str(', '.join(
                                map(str, rouge_scores[d_idx][idx])))))
                        file.write("\n")
                    file.write("\n")

        return (
            all_loss,
            avg_single_bleu,
            avg_bleu,
            avg_single_rouge_1_2_l_be,
            avg_rouge_1_2_l_be,
            avg_best_rouge_1_2_l_be
        )

    def save_model(self, model_dir):
        encoder_path = os.path.join(model_dir, "encoder.ckpt")
        decoder_paths = [
                os.path.join(model_dir, "decoder_{}.ckpt".format(idx))
                for idx in range(self.n_decoders)]
        torch.save(self.encoder, encoder_path)
        for idx, path in enumerate(decoder_paths):
            torch.save(self.decoders[idx], path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        # Get the latest modified model (files or directory)
        encoder_path = os.path.join(model_dir, "encoder.ckpt")
        decoder_paths = [
            os.path.join(model_dir, "decoder_{}.ckpt".format(idx))
            for idx in range(self.n_decoders)]

        loader = True
        if not os.path.exists(encoder_path):
            loader = False
        else:
            encoder = torch.load(encoder_path)
        decoders = []
        for path in decoder_paths:
            if not os.path.exists(path):
                loader = False
            else:
                decoders.append(torch.load(path))

        if not loader:
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.encoder = encoder
            self.decoders = decoders
            print_time_info("Load model from {} successfully".format(
                    model_dir))
