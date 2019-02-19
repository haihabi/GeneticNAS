import os
import torch
import time
import math


def get_batch(source, i, bptt):
    # get_batch subdivides the source dataset into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of dataset is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def rnn_genetic_evaluate(ga, input_model, input_criterion, data_source, ntokens, batch_size, bptt):
    input_model.eval()  # Turn on evaluation mode which disables dropout.
    hidden = input_model.init_hidden(batch_size)
    with torch.no_grad():
        for ind in ga.get_current_generation():
            input_model.set_individual(ind)
            total_loss = 0
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i, bptt)
                output, hidden = input_model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * input_criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
            ga.update_current_individual_fitness(ind, total_loss / (len(data_source) - 1))
    return ga.update_population()


def rnn_evaluate(input_model, input_criterion, data_source, ntokens, batch_size, bptt):
    input_model.eval()  # Turn on evaluation mode which disables dropout.
    hidden = input_model.init_hidden(batch_size)
    with torch.no_grad():
        total_loss = 0
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = input_model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * input_criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train_genetic_rnn(ga, train_data, input_model, input_optimizer, input_criterion, ntokens, batch_size, bptt,
                      grad_clip,
                      log_interval, final):
    # Turn on training mode which enables dropout.
    input_model.train()
    total_loss = 0.
    cur_loss = 0
    start_time = time.time()
    hidden = input_model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        input_optimizer.zero_grad()  # zero old gradients for the next back propgation
        if not final: input_model.set_individual(ga.sample_child())  # updating

        output, hidden = input_model(data, hidden)
        loss = input_criterion(output.view(-1, ntokens), targets)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(input_model.parameters(), grad_clip)
        input_optimizer.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss += total_loss
            elapsed = time.time() - start_time
            print('|  {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(batch, len(train_data) // bptt, elapsed * 1000 / log_interval,
                                                      total_loss / log_interval, math.exp(total_loss / log_interval)))
            total_loss = 0
            start_time = time.time()
    return (bptt * cur_loss) / len(train_data)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    @staticmethod
    def single_batchify(data, bsz, input_device):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the dataset across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(input_device)

    def batchify(self, bsz, device):
        return self.single_batchify(self.train, bsz, device), self.single_batchify(self.valid, bsz,
                                                                                   device), self.single_batchify(
            self.test, bsz, device)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
