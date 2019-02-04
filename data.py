import os
import torch
import torchvision
import torchvision.transforms as transforms
from cnn_utils import Cutout


def get_cifar10(config):
    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    if config.get('cutout'):
        train_transform.transforms.append(Cutout(n_holes=config.get('n_holes'), length=config.get('length')))

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.get('batch_size'),
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.get('batch_size_val'),
                                             shuffle=False, num_workers=4)
    return trainloader, testloader


class BatchIterator(object):
    def __init__(self, data):
        pass


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
        # Starting from sequential dataset, batchify arranges the dataset into columns.
        # For instance, with the alphabet as the sequence and batch size 4, we'd get
        # ┌ a g m s ┐
        # │ b h n t │
        # │ c i o u │
        # │ d j p v │
        # │ e k q w │
        # └ f l r x ┘.
        # These columns are treated as independent by the model, which means that the
        # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
        # batch processing.
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
