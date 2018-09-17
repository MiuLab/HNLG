import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(
        self,
        en_embedding,
        embedding,
        en_vocab_size,
        en_embedding_dim,
        hidden_size,
        n_layers=1,
        cell='GRU',
        bidirectional=False
    ):
        super(EncoderRNN, self).__init__()
        self.vocab_size = en_vocab_size
        self.en_embedding = en_embedding
        self.embedding_dim = en_embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.cell = cell

        """
            self.embedding:
                input: [batch_size, seq_length]
                output: [batch_size, seq_length, embedding_dim]
            self.rnn:
                GRU (with batch_first=True):
                    inputs:
                        input:
                            [batch_size, seq_length, input_size(embedding_dim)]
                        h_0:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]
                    outputs:
                        output:
                            [batch_size, seq_length, \
                                    hidden_size * (bidirectional + 1)]
                        h_n:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]
                LSTM (WITH batch_first=True):
                    inputs:
                        input:
                            [batch_size, seq_length, input_size(embedding_dim)]
                        h_0:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]
                        c_0:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]
                    outputs:
                        output:
                            [batch_size, seq_length, \
                                    hidden_size * (bidirectional + 1)]
                        h_n:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]
                        c_n:
                            [batch_size, n_layers * (bidirectional + 1), \
                                    hidden_size]

        """
        if en_embedding:
            self.embedding = embedding
            if cell == "GRU":
                self.rnn = nn.GRU(
                        self.embedding_dim, hidden_size,
                        n_layers, batch_first=True,
                        bidirectional=bidirectional)
            elif cell == "LSTM":
                self.rnn = nn.LSTM(
                        self.embedding_dim, hidden_size,
                        n_layers, batch_first=True,
                        bidirectiional=bidirectional)
        else:
            if cell == "GRU":
                self.rnn = nn.GRU(
                        self.vocab_size, hidden_size,
                        n_layers, batch_first=True,
                        bidirectional=bidirectional)
            elif cell == "LSTM":
                self.rnn = nn.LSTM(
                        self.vocab_size, hidden_size,
                        n_layers, batch_first=True,
                        bidirectiional=bidirectional)

    def forward(self, inputs, hidden, cell=None):
        if self.en_embedding:
            embedded = self.embedding(inputs)
        else:
            size = inputs.size()
            embedded = torch.Tensor(size[0], size[1], self.vocab_size).zero_()
            inputs = inputs.data.unsqueeze(2)
            embedded.scatter_(dim=2, index=inputs.cpu(), value=1.)
            embedded = Variable(embedded.cuda())
        if self.cell == "GRU":
            output, hidden = self.rnn(embedded, hidden)
            return output, hidden
        elif self.cell == "LSTM":
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            return output, hidden, cell

    def initHidden(self, batch_size):
        result = Variable(
                torch.zeros(
                    self.n_layers * (self.bidirectional + 1),
                    batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def initAttrHidden(self, inputs, batch_size=32):
        # use attributes (encoder input) as RNN initial state
        size = inputs.size()
        attrs = torch.Tensor(size[0], size[1], self.vocab_size).zero_()
        inputs = inputs.data.unsqueeze(2)
        attrs.scatter_(dim=2, index=inputs.cpu(), value=1.)

        # compress attrs into a initial state vector:
        # (batch_size, seq_length, num_attr) -> (batch_size, num_attr)
        attrs = attrs.sum(dim=1)
        # trim _UNK and _PAD
        attrs[:, 0:2] = 0
        # pad zeros
        attrs = torch.cat(
                [
                    attrs,
                    torch.zeros(batch_size, self.hidden_size - self.vocab_size)
                ], 1)
        attrs = attrs.repeat(self.n_layers * (self.bidirectional + 1), 1, 1)
        return Variable(attrs.cuda()) if use_cuda else Variable(attrs)


class Attn(nn.Module):
    def __init__(
            self, method, en_hidden_size, de_hidden_size,
            n_en_layers, n_de_layers, bidirectional):
        super(Attn, self).__init__()

        self.method = method
        self.n_en_layers = n_en_layers
        self.n_de_layers = n_de_layers
        en_hidden_size = en_hidden_size * n_en_layers * (bidirectional + 1)
        de_hidden_size = de_hidden_size * n_de_layers * (bidirectional + 1)

        # If en_layers != de_layers,
        # then the dot attention is same as general attention
        if self.method == "dot" and self.n_en_layers != self.n_de_layers:
            self.method = 'general'

        if self.method == 'general':
            self.attn = nn.Linear(en_hidden_size, de_hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(en_hidden_size + de_hidden_size, 1)

    def forward(self, hidden, encoder_hiddens):
        attn_energies = self.score(hidden, encoder_hiddens)
        return torch.nn.Softmax()(attn_energies)

    def score(self, hidden, encoder_output):
        """
            input:
                hidden: [batch_size, de_hidden_size]
                encoder_output: [batch_size, seq_length, en_hidden_size]
            output:
                score: [batch_size, seq_length]
        """
        if self.method == 'dot':
            encoder_output = encoder_output.permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * encoder_output).sum(1)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output).permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * energy).sum(1)
            return energy

        elif self.method == 'concat':
            seq_length = encoder_output.size(1)
            hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
            energy = self.attn(
                    torch.cat((hidden, encoder_output), 2)).squeeze(2)
            return energy


class HAttn(nn.Module):
    def __init__(
            self, method, en_hidden_size, de_hidden_size,
            n_en_layers, n_de_layers, bidirectional):
        super(HAttn, self).__init__()

        self.method = method
        self.n_en_layers = n_en_layers
        self.n_de_layers = n_de_layers
        de_hidden_size = de_hidden_size * n_de_layers * (bidirectional + 1)

        # If en_layers != de_layers,
        # then the dot attention is same as general attention
        if self.method == "h_dot" and self.n_en_layers != self.n_de_layers:
            self.method = 'h_general'

        if self.method == 'h_general':
            self.attn = nn.Linear(de_hidden_size, de_hidden_size)

        elif self.method == 'h_concat':
            self.attn = nn.Linear(de_hidden_size + de_hidden_size, 1)
        elif self.method == 'h_concat2':
            self.attn = nn.Sequential(
                nn.Linear(de_hidden_size + de_hidden_size, de_hidden_size),
                nn.Tanh(),
                nn.Linear(de_hidden_size, 1, bias=False)
            )

    def forward(self, hidden, encoder_hiddens):
        attn_energies = self.score(hidden, encoder_hiddens)
        return torch.nn.Softmax()(attn_energies)

    def score(self, hidden, encoder_output):
        """
            input:
                hidden: [batch_size, de_hidden_size]
                encoder_output: [batch_size, seq_length, en_hidden_size]
            output:
                score: [batch_size, seq_length]
        """
        if self.method == 'h_dot':
            encoder_output = encoder_output.permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * encoder_output).sum(1)
            return energy

        elif self.method == 'h_general':
            energy = self.attn(encoder_output).permute(0, 2, 1)
            hidden = hidden.unsqueeze(2)
            energy = (hidden * energy).sum(1)
            return energy

        elif self.method == 'h_concat':
            seq_length = encoder_output.size(1)
            hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
            energy = self.attn(
                    torch.cat((hidden, encoder_output), 2)).squeeze(2)
            return energy
        elif self.method == 'h_concat2':
            seq_length = encoder_output.size(1)
            hidden = hidden.unsqueeze(1).repeat(1, seq_length, 1)
            energy = self.attn(
                    torch.cat((hidden, encoder_output), 2)).squeeze(2)
            return energy


class DecoderRNN(nn.Module):
    def __init__(
            self,
            embedding,
            de_vocab_size,
            de_embedding_dim,
            en_hidden_size,
            de_hidden_size,
            n_en_layers=1,
            n_de_layers=1,
            cell='GRU',
            attn_method='concat',
            bidirectional=False,
            feed_last=False,
            batch_norm=False,
            h_attn=False,
            index=0
    ):
        super(DecoderRNN, self).__init__()
        self.vocab_size = de_vocab_size
        self.embedding_dim = \
            de_embedding_dim * 2 if feed_last else de_embedding_dim
        self.hidden_size = de_hidden_size
        self.n_layers = n_de_layers
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        self.cell = cell
        """
            self.embedding:
                input: [batch_size, seq_length]
                output: [batch_size, seq_length, embedding_dim]
            self.rnn:
                GRU (with batch_first=True):
                    inputs:
                        input:
                            [batch_size, seq_length, rnn_dim]
                        h_0:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]
                    outputs:
                        output:
                            [batch_size, seq_length, \
                                    hidden_size * bidirectional]
                        h_n:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]
                LSTM (WITH batch_first=True):
                    inputs:
                        input:
                            [batch_size, seq_length, rnn_dim]
                        h_0:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]
                        c_0:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]
                    outputs:
                        output:
                            [batch_size, seq_length, \
                                    hidden_size * (bidirectional + 1)]
                        h_n:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]
                        c_n:
                            [n_layers * (bidirectional + 1), batch_size, \
                                    hidden_size]

        """
        self.h_attn = h_attn
        if attn_method != 'none':
            rnn_dim = self.embedding_dim + en_hidden_size * (bidirectional + 1)
            self.attn = Attn(
                    attn_method, en_hidden_size, de_hidden_size,
                    n_en_layers, n_de_layers, bidirectional)

            if self.h_attn:
                if index != 0:
                    rnn_dim = self.embedding_dim + de_hidden_size * (bidirectional + 1)
                    self.attn = HAttn(
                        attn_method, en_hidden_size, de_hidden_size,
                        n_en_layers, n_de_layers, bidirectional
                    )
                else:
                    rnn_dim = self.embedding_dim
                    self.attn = None

        else:
            rnn_dim = self.embedding_dim
            self.attn = None

        self.embedding = embedding
        self.cell = cell

        if cell == "GRU":
            self.rnn = nn.GRU(
                    rnn_dim, de_hidden_size, n_de_layers,
                    batch_first=True, bidirectional=bidirectional)
        elif cell == "LSTM":
            self.rnn = nn.LSTM(
                    rnn_dim, de_hidden_size, n_de_layers,
                    batch_first=True, bidirectiional=bidirectional)

        # to handle encoder decoder hidden_size mismatch
        self.transform_layer = nn.Linear(en_hidden_size, de_hidden_size)

        if bidirectional:
            self.out = nn.Linear(de_hidden_size * 2, self.vocab_size)
        else:
            self.out = nn.Linear(de_hidden_size, self.vocab_size)

        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(
                    de_hidden_size * (1 + bidirectional))

        self.feed_last = feed_last

    def forward(
            self, inputs, last_hidden, encoder_hiddens,
            last_cell=None, last_output=None, last_decoder_hiddens=None):
        """
            last_hidden:
                [n_layers * (bidirectional + 1), batch_size, hidden_size]
            encoder_hiddens:
                [batch_size, seq_length, hidden_size * (bidirectional + 1)]
            last_output:
                [batch_size, seq_length]
        """
        embedded = self.embedding(inputs)
        if self.feed_last and last_output is not None:
            embedded = torch.cat((embedded, self.embedding(last_output)), 2)
        if self.attn:
            batch_size = last_hidden.size(1)
            if not self.h_attn:
                attn_weights = self.attn(
                    last_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1),
                    encoder_hiddens).unsqueeze(2)
                attn = (attn_weights * encoder_hiddens).sum(1).unsqueeze(1)
            elif self.h_attn:
                attn_weights = self.attn(
                    last_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1),
                    last_decoder_hiddens
                )
                attn = (attn_weights.unsqueeze(2) * last_decoder_hiddens).sum(1).unsqueeze(1)
            rnn_input = torch.cat((embedded, attn), 2)
        else:
            rnn_input = embedded
        """
            use CrossEntropy Loss, output logit
            LogSoftmax + NLLLoss:
            output = self.log_softmax(self.out(output))
            maybe more stable operation by adding a little value to softmax
            probs before log operation
            output = self.softmax(self.out(output).squeeze(1)) \
                    .add(1e-6).log().div(np.log(10)).unsqueeze(1)
        """
        if self.cell == "GRU":
            output, hidden = self.rnn(rnn_input, last_hidden)
            if self.batch_norm:
                output = self.batch_norm(output.squeeze(1))
                output = self.out(output).unsqueeze(1)
            else:
                output = self.out(output)
            return output, hidden
        elif self.cell == "LSTM":
            output, (hidden, cell) = self.rnn(
                    embedded, (last_hidden, last_cell))
            if self.batch_norm:
                output = self.batch_norm(output.squeeze(1))
                output = self.out(output).unsqueeze(1)
            else:
                output = self.out(output)
            return output, hidden, cell

    def initHidden(self, batch_size):
        result = Variable(
                torch.zeros(
                    self.n_layers * (self.bidirectional + 1),
                    batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
