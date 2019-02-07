from params import read_args
from pickel_data import loading_variable
from ultis import shuffled_mini_batches, load_info_label
import torch
import os
import datetime
# from PatchNet_CNN_advanced import PatchNet
# from PatchNet_CNN import PatchNet
from PatchNet_CNN_fast import PatchNet
import torch.nn as nn
from evaluation import eval


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
            pad_msg, pad_added_code, pad_removed_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                    pad_added_code).long(), torch.tensor(pad_removed_code).long(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
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


def train_all_model(commits, params):
    labels, pad_msg, pad_added_code, pad_removed_code, dict_msg, dict_code = commits

    print('Number of commits for training model: %i' % (len(labels)))
    print('Commit message dictionary: %i' % (len(dict_msg)))
    print('Commit code dictionary: %i' % (len(dict_code)))
    print('Pos -- Neg -- ', load_info_label(data=labels))

    batches = shuffled_mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code,
                                    Y=labels, mini_batch_size=params.batch_size)
    print('Number of batches: %i' % (len(batches)))
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_all_model(batches=batches, model=model, params=params)


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_all_model(commits=loading_variable('train_all_recent_sasha'), params=input_option)
