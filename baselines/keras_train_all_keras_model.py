from params import read_args
from pickel_data import loading_variable


def train_all_model(commits, params):
    print(len(commits))


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    train_all_model(commits=loading_variable('train_all_recent_keras'), params=input_option)