from params import read_args
from pickel_data import loading_variable
from keras_model import lstm_cnn_all_data

def load_info_label(data):
    data = list(data)
    pos = [1 for d in data if d == 1]
    neg = [0 for d in data if d == 0]
    return len(pos), len(neg)


def train_all_model(commits, params):
    labels, pad_, dict_ = commits

    print('Number of commits for training model: %i' % (len(labels)))
    print('Dictionary: %i' % (len(dict_)))
    print('Pos -- Neg -- ', load_info_label(data=labels))
    params.vocab_msg = len(dict_)
    lstm_cnn_all_data(x_train=pad_, y_train=labels, FLAGS=params)


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_all_model(commits=loading_variable('train_all_recent_keras'), params=input_option)