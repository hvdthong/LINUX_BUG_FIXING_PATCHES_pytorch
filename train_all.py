from params import read_args
from pickel_data import loading_variable


def train_all_model(train_commits, params):
    print(len(train_commits))
    print('hello')


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    train_all_model(train_commits=loading_variable('train_all'), params=input_option)