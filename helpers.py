import torch
import numpy as np
from torch.autograd import Variable
from math import ceil
import random


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def batchwise_sample(gen, num_samples, batch_size, seq_len):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size, seq_len))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
    
def load_music_file(file):
    with open(file, 'r') as file:
        text = file.read()
        
    # get vocabulary set
    words = sorted(tuple(set(text.split())))
    n = len(words)

    # create word-integer encoder/decoder
    word2int = dict(zip(words, list(range(n))))
    int2word = dict(zip(list(range(n)), words))

    # encode all words in dataset into integers
    encoded = np.array([word2int[word] for word in text.split()])
    
    return n, word2int, int2word, encoded

def batch_music_samples(data, seq_len):
    batches = []
    i=0
    while i+seq_len <= len(data):
        batch = list(data[i:i+seq_len])
        batches.append(batch)
        i += seq_len
    return batches
    
def train_val_split(data, train_size):
    random.seed(1)
    random.shuffle(data)
    train = torch.tensor(data[:train_size]).type(torch.LongTensor)
    val = torch.tensor(data[train_size:]).type(torch.LongTensor)
    return train, val

def positive_sample(data, n):
    torch.tensor(data)
    sample_idx = random.sample(range(1, len(data)), n)
    return data[sample_idx]   
    