from keras_ultis import extract_commit, reformat_commit_code_keras, extract_commit_recent_sasha, \
    extract_commit_recent_sasha_keras
from keras_padding import padding_commit
from params import read_args
from pickel_data import saving_variable


if __name__ == '__main__':
    # creating pickle data for training and testing
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # loading recent sasha data
    path_file = '../data/recent_sashas_functions_translated_nodups.out'
    commits_recent_sasha = extract_commit_recent_sasha_keras(path_file=path_file)
    commits_recent_sasha = reformat_commit_code_keras(commits=commits_recent_sasha, num_file=1)

    # loading training data
    path_file = '../data/newres_funcalls_jul28.out.sorted'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code_keras(commits=commits, num_file=1)

    # add recent sasha data with training data
    commits_all = commits + commits_recent_sasha
    train_all = padding_commit(commits=commits_all, params=input_option)
    saving_variable('train_all_recent_keras', variable=train_all)


