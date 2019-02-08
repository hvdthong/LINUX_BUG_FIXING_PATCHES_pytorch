from params import read_args
from pickel_data import loading_variable
from ultis import shuffled_mini_batches, load_info_label
import torch
import os
import datetime
from CNN_pytorch import CNN
import torch.nn as nn
from evaluation import eval
import numpy as np
import math


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def running_all_model(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches:
            pad_msg_code, labels = batch
            if torch.cuda.is_available():
                pad_msg_code, labels = torch.tensor(pad_msg_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg_code, labels = torch.tensor(pad_msg_code).long(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))
        print('Epoch: %i / %i ---Data' % (epoch, params.num_epochs))
        acc, prc, rc, f1, auc_ = eval(data=batches, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


def shuffled_mini_batches(X_, Y, mini_batch_size=64, seed=0):
    m = X_.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_ = X_[permutation, :]

    if len(Y.shape) == 1:
        shuffled_Y = Y[permutation]
    else:
        shuffled_Y = Y[permutation, :]
    # shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_ = shuffled_X_[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_ = shuffled_X_[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def train_all_model(commits, params):
    labels, pad_, dict_ = commits

    print('Number of commits for training model: %i' % (len(labels)))
    print('Dictionary: %i' % (len(dict_)))
    print('Pos -- Neg -- ', load_info_label(data=labels))

    batches = shuffled_mini_batches(X_=pad_, Y=labels, mini_batch_size=params.batch_size)
    print('Number of batches: %i' % (len(batches)))
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%lY-%m-%d_%H-%M-%S'))
    params.vocab_msg = len(dict_)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_all_model(batches=batches, model=model, params=params)


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_all_model(commits=loading_variable('train_all_recent_keras'), params=input_option)
