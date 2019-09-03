from keras_ultis import extract_commit, reformat_commit_code_keras, extract_commit_recent_sasha_keras, \
    extract_commit_recent_sasha_keras_ver1
from keras_padding import padding_train_test_commits
from keras_pickel_data import saving_variable
from params import read_args

if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    path_file = '../data/recent_sashas_functions_translated_nodups.out'
    commits_recent_sasha = extract_commit_recent_sasha_keras(path_file=path_file)
    commits_recent_sasha = reformat_commit_code_keras(commits=commits_recent_sasha, num_file=1)

    # loading training data
    path_file = '../data/newres_funcalls_jul28.out.sorted'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code_keras(commits=commits, num_file=1)

    # add recent sasha data with training data
    train_commits = commits + commits_recent_sasha

    # loading recent linux data
    # path_file = '../data/recent_functions_translated.out'
    path_file = '../data/recent_functions_translated_ver1.out'
    commits_recent_data = extract_commit_recent_sasha_keras_ver1(path_file=path_file)
    commits_recent_data = reformat_commit_code_keras(commits=commits_recent_data, num_file=1)
    test_commits = commits_recent_data
    print(len(train_commits), len(test_commits))

    train_data, test_data, dict_ = padding_train_test_commits(train=train_commits, test=test_commits,
                                                              params=input_option)
    data = (train_data, test_data, dict_)
    # saving_variable('train_all_recent_new_data_unknown_labels', variable=data)
    saving_variable('train_all_recent_new_data_unknown_labels_ver1', variable=data)
