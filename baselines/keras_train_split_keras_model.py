from params import read_args
from keras_pickel_data import loading_variable
from keras_model import lstm_cnn_split_data


def load_info_label(data):
    data = list(data)
    pos = [1 for d in data if d == 1]
    neg = [0 for d in data if d == 0]
    return len(pos), len(neg)


def train_split_model(commits, params):
    train_data, test_data, dict_ = commits
    train_labels, train_commits = train_data
    test_labels, test_commits = test_data

    print('Shape of training data:', (len(train_commits.shape)))
    print('Shape of testing data:', (len(test_commits.shape)))
    print('Dictionary: %i' % (len(dict_)))
    print('Training: Pos -- Neg -- ', load_info_label(data=train_labels))
    print('Testing: Pos -- Neg -- ', load_info_label(data=test_labels))
    print(train_commits.shape, test_commits.shape)

    params.vocab_msg = len(dict_)
    lstm_cnn_split_data(x_train=train_commits, y_train=train_labels, x_test=test_commits, y_test=test_labels, FLAGS=params)


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_split_model(commits=loading_variable('train_split_recent_keras'), params=input_option)
