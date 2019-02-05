import gnas
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EmbeddingDropout(torch.nn.Embedding):
    """Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.

    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).

    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.1,
                 scale=None):
        """Embedding constructor.

        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.

        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), ('Dropout must be >= 0.0 '
                                                      'and < 1.0')
        self.scale = scale

    def forward(self, inputs):  # pylint:disable=arguments-differ
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)


class LockedDropout(nn.Module):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # The input is of size [T,N,F] where T is the number of steps, N is the batch size
        # and F is the number of features.
        # The random variable generate for the locked dropout don't change over the T axis
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, ss=None):
        super(RNNModel, self).__init__()
        self.drop_input = LockedDropout(0.65)  # TODO:change to input config
        self.drop_end = LockedDropout(0.4)  # TODO:change to input config
        self.encoder = EmbeddingDropout(ntoken, ninp, dropout=0.1)  # TODO:change to input config
        self.ss = ss
        self.rnn = gnas.modules.RnnSearchModule(in_channels=ninp, n_channels=nhid,
                                                working_device='cuda',
                                                ss=self.ss)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def set_individual(self, individual):
        self.rnn.set_individual(individual)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop_input(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop_end(output)
        decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        return decoded.contiguous().view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return self.rnn.init_state(bsz)
