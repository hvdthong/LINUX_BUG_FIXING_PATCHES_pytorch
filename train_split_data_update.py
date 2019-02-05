from pickel_data import loading_variable
from ultis import mini_batches
from params import read_args
import torch
import os, datetime
# from PatchNet_CNN import PatchNet
from PatchNet_CNN_advanced import PatchNet
import torch.nn as nn
from evaluation import eval
from train_split_data import save


def running_split_model_update(train_batches, test_batches, model, params):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(params.file_model))
    else:
        model.load_state_dict(torch.load(params.file_model, map_location='cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in train_batches:
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

        print('Epoch: %i ---Training data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=train_batches, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        print('Epoch: %i ---Testing data' % (epoch))
        acc, prc, rc, f1, auc_ = eval(data=test_batches, model=model)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1



def train_split_model_update(train_commits, test_commits, train_dict, params):
    train_labels, train_msg, train_added_code, train_removed_code = train_commits
    test_labels, test_msg, test_added_code, test_removed_code = test_commits

    print('Number of commits for training model: %i' % (len(train_labels)))
    print('Number of commits for testing: %i' % (len(test_labels)))

    train_batches = mini_batches(X_msg=train_msg, X_added_code=train_added_code, X_removed_code=train_removed_code,
                                 Y=train_labels, mini_batch_size=params.batch_size)
    test_batches = mini_batches(X_msg=test_msg, X_added_code=test_added_code, X_removed_code=test_removed_code,
                                Y=test_labels, mini_batch_size=params.batch_size)
    print(len(train_batches), len(test_batches))

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    dict_msg, dict_code = train_dict
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(train_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = train_labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchNet(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_split_model_update(train_batches=train_batches, test_batches=test_batches, model=model, params=params)


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    train_split_model_update(train_commits=loading_variable('train'), test_commits=loading_variable('test'),
                             train_dict=loading_variable('train_dict'), params=input_option)
