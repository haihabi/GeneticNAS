# coding: utf-8
import argparse
import time
import torch.nn as nn
import torch.onnx
from torch import optim
import data
import model
import gnas
from rnn_utils import train_genetic_rnn, rnn_genetic_evaluate
from gnas.genetic_algorithm.annealing_functions import cosine_annealing

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the dataset corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--save_auto', type=str, default='model_auto.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load dataset
###############################################################################
corpus = data.Corpus(args.data)
eval_batch_size = args.batch_size
train_data, val_data, test_data = corpus.batchify(args.batch_size, device)
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
ss = gnas.get_enas_rnn_search_space(args.emsize, args.nhid, 12)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, ss=ss).to(
    device)
model.set_individual(ss.generate_individual())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), weight_decay=0.00000001,
                      lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

###############################################################################
# Training code
###############################################################################
# Loop over epochs.
lr = args.lr
best_val_loss = None
enable_search = True
# At any point you can hit Ctrl + C to break out of training early.
try:
    ga = gnas.genetic_algorithm_searcher(ss, population_size=20)
    for epoch in range(1, args.epochs + 1):
        if epoch > 15:
            scheduler.step()
        epoch_start_time = time.time()
        p = cosine_annealing(epoch, 1, 15, 125)
        train_loss = train_genetic_rnn(ga, train_data, p, model, optimizer, criterion, ntokens, args.batch_size,
                                       args.bptt, args.clip,
                                       args.log_interval)
        val_loss, loss_var, max_loss, min_loss = rnn_genetic_evaluate(ga, model, criterion, val_data, ntokens,
                                                                      eval_batch_size, args.bptt)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | lr {:02.2f} |  '
              ''.format(epoch, (time.time() - epoch_start_time),
                        val_loss, scheduler.get_lr()[-1]))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
    ga.save_result('')
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#     # after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
#     model.rnn.flatten_parameters()

# Run on test dataset.
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
