from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import music_helpers

import random


CUDA = False
# VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000   # consider increasing to 500k
N_SAMPLES = 5

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

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

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            
            inp, target = music_helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        print(' average_train_NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        
        s = gen.sample(BATCH_SIZE*2)        # 64 works best, try large to leverage more GPU
        inp, target = music_helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)
        
        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, real_val, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    
    pos_val = positive_sample(real_val, 100)
    neg_val = generator.sample(100)
    val_inp, val_target = music_helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        
        s = music_helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        
        dis_inp, dis_target = music_helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':
    
    # prepare train and val data
    mozart_data = "./data/mozart.txt"
    VOCAB_SIZE, word2int, int2word, encoded_data = load_music_file(mozart_data)
    real_data_samples = batch_music_samples(encoded_data, MAX_SEQ_LEN)
    real_train, real_val = train_val_split(real_data_samples, POS_NEG_SAMPLES)

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()
        real_train = real_train.cuda()
        real_val = real_val.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                            start_letter=START_LETTER, gpu=CUDA)
    # print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator: ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, dis, 1)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, real_train, gen, real_val, 5, 3)

        # generate output samples
        samples20 = gen.sample(N_SAMPLES)

        for i in range(N_SAMPLES):
            sample20 = ' '.join([int2word[x] for x in list(np.array(samples20[i]))])        
            with open("./data/output/result_epoch" + str(epoch) + "_sample" + str(i) + "_20.txt", "w") as outfile:
                    outfile.write(sample20)
                
        